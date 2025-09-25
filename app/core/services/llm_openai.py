"""
Purpose: Thin client wrapper around OpenAI (or other LLMs later).
One place for auth, retries, model options, response/usage normalization.

Extensibility:
- Add other providers (AnthropicLLM, LocalLLM) without touching controller.
- Add streaming support later (yield tokens) behind the same interface.

Testing: Mock SDK calls; assert it maps usage and errors correctly.
"""

from __future__ import annotations
import time
from typing import Optional
from ..models import LLMSettings

try:
    from openai import OpenAI
    from openai import APIError, RateLimitError, APITimeoutError
except Exception:
    OpenAI = None
    APIError = RateLimitError = APITimeoutError = AuthenticationError = Exception


class OpenAILLMClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        if not self.api_key:
            raise RuntimeError("Missing OPENAI_API_KEY")
        if OpenAI is None:
            raise RuntimeError("openai package not installed. pip install openai")
        try:
            self.client = OpenAI(api_key=self.api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

    def _with_retries(self, fn, *args, **kwargs):
        for delay in [0.5, 1.0, 2.0, 4.0]:
            try:
                return fn(*args, **kwargs)
            except (RateLimitError, APITimeoutError, APIError):
                time.sleep(delay)
        return fn(*args, **kwargs)

    def chat(
        self,
        messages: list[dict[str, str]],
        settings: LLMSettings,
        system: Optional[str] = None,
    ):
        payload = []
        if system:
            payload.append({"role": "system", "content": system})
        payload.extend(messages)

        try:
            if hasattr(self.client, "responses"):

                def call_resp():
                    return self.client.responses.create(
                        model=settings.model,
                        input=[
                            {"role": m["role"], "content": m["content"]}
                            for m in payload
                        ],
                        temperature=settings.temperature,
                        top_p=settings.top_p,
                        max_output_tokens=settings.max_tokens,
                        frequency_penalty=settings.frequency_penalty,
                        presence_penalty=settings.presence_penalty,
                        response_format=settings.response_format or None,
                    )

                resp = self._with_retries(call_resp)
                text = getattr(resp, "output_text", None)
                usage = getattr(resp, "usage", None)
                tokens_in = getattr(usage, "input_tokens", 0) if usage else 0
                tokens_out = getattr(usage, "output_tokens", 0) if usage else 0
                if text:
                    return text, {
                        "model": resp.model,
                        "tokens_in": tokens_in,
                        "tokens_out": tokens_out,
                        "raw": resp,
                    }
        except Exception:
            pass

        def call_cc():
            return self.client.chat.completions.create(
                model=settings.model,
                messages=payload,
                temperature=settings.temperature,
                top_p=settings.top_p,
                max_tokens=settings.max_tokens,
                frequency_penalty=settings.frequency_penalty,
                presence_penalty=settings.presence_penalty,
            )

        cc = self._with_retries(call_cc)
        text = cc.choices[0].message.content
        usage = getattr(cc, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", 0) if usage else 0
        tokens_out = getattr(usage, "completion_tokens", 0) if usage else 0
        return text, {
            "model": cc.model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "raw": cc,
        }

    def image_generate(self, *, prompt: str, size: str = "auto", n: int = 1):
        """
        Generate images from text prompt using OpenAI's image generation API.
        Tokens in and out are just a rought estimates."""
        resp = self.client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size=size,
            n=n,
        )

        url = getattr(resp.data[0], "url", None)
        b64 = getattr(resp.data[0], "b64_json", None)

        meta = {
            "tokens_in": len(prompt.split()),
            "tokens_out": 6240 * n,
            "model": "gpt-image-1",
        }
        if url:
            return {"kind": "url", "data": url}, meta

        if b64:
            import base64

            png_bytes = base64.b64decode(b64)
            return {"kind": "bytes", "data": png_bytes, "format": "PNG"}, meta

        return {"kind": "none", "data": None}, meta
