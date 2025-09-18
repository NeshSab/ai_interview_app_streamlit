from .services.llm_openai import OpenAILLMClient
from .prompts.roadmap import build_infographic_prompt


class RoadmapController:
    def __init__(self, llm: OpenAILLMClient | None = None):
        self.llm = llm or OpenAILLMClient()
        self.tokens_in = 0
        self.tokens_out = 0
        self.model_used = None

    def generate_infographic(self, plan: dict, *, size: str = "1024x1536") -> str:
        prompt = build_infographic_prompt(plan)
        url, meta = self.llm.image_generate(prompt=prompt, size=size, n=1)

        self.tokens_in += int(meta.get("tokens_in", 0))
        self.tokens_out += int(meta.get("tokens_out", 0))
        self.model_used = meta.get("model")

        return url, meta
