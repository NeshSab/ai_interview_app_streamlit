"""
Purpose: text-to-speech integration. Allow voice-based practice and read-outs.
"""

from __future__ import annotations
import os
import tempfile
from ..interfaces import LLMClient


def tts_bytes(
    text: str,
    llm: LLMClient,
    *,
    voice: str = "alloy",
    model: str = "gpt-4o-mini-tts",
    max_chars: int = 1200,
) -> bytes:
    """
    Return raw MP3 bytes. Tries streaming path; falls back to non-streaming.
    """
    safe = (text or "").strip()
    if not safe:
        return b""
    if len(safe) > max_chars:
        safe = safe[: max_chars - 1].rstrip() + "…"

    client = getattr(llm, "client", llm)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp_path = tmp.name
        try:
            with client.audio.speech.with_streaming_response.create(
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
        pass

    resp = client.audio.speech.create(model=model, voice=voice, input=safe)
    if hasattr(resp, "to_bytes"):
        return resp.to_bytes()
    if hasattr(resp, "content"):
        return resp.content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp2:
        tmp2_path = tmp2.name
    try:
        if hasattr(resp, "stream_to_file"):
            resp.stream_to_file(tmp2_path)
            with open(tmp2_path, "rb") as f:
                return f.read()
    finally:
        try:
            os.remove(tmp2_path)
        except Exception:
            pass

    return b""
