"""
Purpose: STT/TTS integration.
Why: Allow voice-based practice, audio prompts, and read-outs.

What is inside (future):
- Transcriber.transcribe(audio_bytes) -> str
- SpeechSynthesizer.speak(text) -> audio_bytes
- Plug into core.interfaces for easy swapping (e.g., Whisper, ElevenLabs, etc.).

Testing: Use canned audio fixtures; assert latency/size limits.
"""

# core/services/speech.py
from __future__ import annotations
import os
import tempfile
from typing import Optional

try:
    from openai import OpenAI  # v1 SDK
except Exception:
    OpenAI = None


def _get_client(timeout: float = 60.0):
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not available")
    # timeout applies to each request
    return OpenAI(timeout=timeout)


def tts_bytes(
    text: str,
    *,
    voice: str = "alloy",
    model: str = "gpt-4o-mini-tts",
    client: Optional[object] = None,
    max_chars: int = 1200,  # safety cap; huge prompts slow TTS
) -> bytes:
    """
    Return raw MP3 bytes. Tries streaming path; falls back to non-streaming.
    """
    safe = (text or "").strip()
    if not safe:
        return b""
    if len(safe) > max_chars:
        safe = safe[: max_chars - 1].rstrip() + "…"

    cli = client or _get_client()

    # Try streaming path first
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp_path = tmp.name
        try:
            # Newer SDKs expose .with_streaming_response
            with cli.audio.speech.with_streaming_response.create(
                model=model, voice=voice, input=safe
            ) as resp:
                resp.stream_to_file(tmp_path)
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    except AttributeError:
        # Fall back to non-streaming path
        pass

    # Non-streaming fallback
    resp = cli.audio.speech.create(model=model, voice=voice, input=safe)
    # In v1 SDK, resp is a Speech object with .to_bytes() or .content
    if hasattr(resp, "to_bytes"):
        return resp.to_bytes()
    if hasattr(resp, "content"):
        return resp.content  # type: ignore[attr-defined]
    # Last resort: stream to temp via export
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp2:
        tmp2_path = tmp2.name
    try:
        # Some SDK builds expose .stream_to_file even on non-streaming
        if hasattr(resp, "stream_to_file"):
            resp.stream_to_file(tmp2_path)  # type: ignore[attr-defined]
            with open(tmp2_path, "rb") as f:
                return f.read()
    finally:
        try:
            os.remove(tmp2_path)
        except Exception:
            pass

    # Couldn’t find bytes container; give up
    return b""
