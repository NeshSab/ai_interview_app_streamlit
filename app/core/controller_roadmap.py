"""
Controller for generating infographic images from plans.
Uses an LLM client and prompt factory to create prompts and generate images.
"""

from .prompts.factory import DefaultPromptFactory
from .interfaces import PromptFactory, LLMClient
from typing import Optional


class RoadmapController:
    def __init__(
        self,
        llm: LLMClient,
    ):
        self.llm: LLMClient = llm
        self.prompts: PromptFactory = DefaultPromptFactory()
        self.tokens_in: int = 0
        self.tokens_out: int = 0
        self.model_used: Optional[str] = None

    def generate_infographic(self, plan: dict, *, size: str = "1024x1024") -> str:
        """Generates an infographic image from a plan dictionary."""
        prompt = self.prompts.build_infographic_prompt(plan)
        url, meta = self.llm.image_generate(prompt=prompt, size=size, n=1)

        self.tokens_in += int(meta.get("tokens_in", 0))
        self.tokens_out += int(meta.get("tokens_out", 0))
        self.model_used = meta.get("model")

        return url, meta
