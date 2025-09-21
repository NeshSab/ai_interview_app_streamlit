"""
Purpose: speech-to-text integration. Allow voice-based inputs.
"""

from __future__ import annotations
import io
import base64
import uuid
from ..interfaces import LLMClient


def transcribe_wav_bytes(
    wav_bytes: bytes, llm: LLMClient, *, model: str = "whisper-1"
) -> str:
    """
    Transcribe WAV audio bytes to text using the given LLM client (e.g., OpenAI)."""
    client = getattr(llm, "client", llm)
    with io.BytesIO(wav_bytes) as buf:
        buf.name = "input.wav"
        resp = client.audio.transcriptions.create(model=model, file=buf)
    return (resp.text or "").strip()


def autoplay_html(mp3_bytes: bytes) -> str:
    """Return an HTML snippet that auto-plays MP3 bytes."""
    if not mp3_bytes:
        return ""
    b64 = base64.b64encode(mp3_bytes).decode("ascii")
    el_id = f"tts_{uuid.uuid4().hex}"
    return f"""
    <audio id="{el_id}" autoplay playsinline preload="auto" style="display:none">
      <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
    </audio>
    <script>
      (function() {{
        const a = document.getElementById("{el_id}");
        if (a) {{
          a.play().catch(() => {{
            // Autoplay blocked → we silently ignore; UI toggle/button acts 
            as user gesture later
          }});
        }}
      }})();
    </script>
    """
