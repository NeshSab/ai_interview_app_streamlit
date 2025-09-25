"""
Abstractions for pluggable services. Inversion of control—core depends
on interfaces, not concrete services. Enables fakes/mocks and future swaps.
Protocols define what services or components can do,
without saying how they do it.

Common protocols:
- LLMClient.chat(messages, settings) -> (reply, meta)
- PromptFactory.build_system(...)-> str & assemble(...) -> List[Message]
- SecurityGuard.validate_user_input(text) / validate_job_description(jd)

Testing: Use simple fake implementations to test the controller without network calls.
"""

from __future__ import annotations
from typing import Optional, Protocol
from .models import Message, LLMSettings, JobDescription


class LLMClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        settings: LLMSettings,
        system: Optional[str] = None,
    ) -> tuple[str, dict]: ...

    def image_generate(self, *, prompt: str, size: str = "auto", n: int = 1): ...


class PromptFactory(Protocol):

    def build_system(
        self,
        *,
        persona,
        style,
        role: str,
        seniority: str,
        mode,
        interviewer_role,
        jd: JobDescription | None,
        memos=None,
    ) -> str: ...

    def assemble(
        self, *, system: str, history: list[Message], user_text: str
    ) -> list[Message]: ...

    def greeting_instruction(self) -> str: ...

    def build_feedback_system(
        self,
        *,
        persona,
        style,
        role: str,
        seniority: str,
        mode,
        interviewer_role,
        jd: JobDescription | None,
        memos=None,
    ) -> str: ...

    def scoring_instruction(self, *, question: str, answer: str) -> str: ...

    def improve_answer_instruction(self, *, question: str, answer: str) -> str: ...

    def build_handoff_system(
        self,
        *,
        persona,
        style,
        role: str,
        seniority: str,
        mode,
        interviewer_role,
        jd: JobDescription | None,
        memos=None,
    ) -> str: ...

    def handoff_instruction(
        self,
        *,
        transcript: str,
        interviewer_role,
        interviewer_persona,
    ) -> str: ...

    def build_jd_parse_system(self) -> str: ...

    def jd_parse_instruction(self, *, text: str) -> str: ...

    def jd_requirements_only_instruction(self, *, role: str, seniority: str) -> str: ...

    def build_role_validation_system(self) -> str: ...

    def validate_role_seniority_instruction(
        self, *, role_text: str, seniority_text: str
    ) -> str: ...

    def build_plan_system(self) -> str: ...

    def jd_plan_with_requirements_instruction(
        self, *, role: str, seniority: str, requirements: list[str]
    ) -> str: ...

    def build_infographic_prompt(self, plan: dict) -> str: ...


class SecurityGuard(Protocol):
    def validate_user_input(self, text: str) -> None: ...

    def validate_job_description_length(self, text: str) -> str: ...

    def validate_role_and_seniority(
        self,
        role_text: str,
        seniority_text: str,
        *,
        llm: Optional[LLMClient] = None,
        model: str = "gpt-4o-mini",
    ) -> tuple[bool, str, bool, str, str]: ...

    def sanitize_for_prompt(self, text: str) -> str: ...

    def redact_pii(self, text: str) -> tuple[str, list[str]]: ...

    def check_prompt_injection(self, text: str) -> None: ...

    def moderate(self, text: str) -> None: ...
