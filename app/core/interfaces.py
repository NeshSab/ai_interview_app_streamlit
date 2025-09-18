"""
(Protocols)
Purpose: Abstractions for pluggable services.
Why: Inversion of control—core depends on interfaces, not concrete services.
Enables fakes/mocks and future swaps.

Common protocols:
- LLMClient.chat(messages, settings) -> (reply, meta)
- PromptFactory.build_system(...)-> str & assemble(...) -> List[Message]
- SecurityGuard.validate_user_input(text) / validate_job_description(jd)
- (Optional) QuestionGenerator, AnswerCritic, SpeechSynthesizer, Transcriber.

Testing: Use simple fake implementations to test the controller without network calls.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Protocol
from .models import Message, LLMSettings, JobDescription, PracticeMode


class LLMClient(Protocol):
    def chat(
        self,
        messages: List[Dict[str, str]],
        settings: LLMSettings,
        system: Optional[str] = None,
    ) -> Tuple[str, Dict]: ...


class PromptFactory(Protocol):
    def build_system(
        self,
        *,
        role: str,
        seniority: str,
        jd: JobDescription | None,
    ) -> str: ...

    def assemble(
        self, *, system: str, history: List[Message], user_text: str
    ) -> List[Message]: ...

    def greeting_instruction(self, mode: Optional[PracticeMode] = None) -> str: ...


class SecurityGuard(Protocol):
    def validate_user_input(self, text: str) -> None: ...

    def validate_job_description(self, jd: JobDescription) -> None: ...

    def validate_role_and_seniority(
        self,
        role_text: str,
        seniority_text: str,
        *,
        llm: Optional[LLMClient] = None,
        model: str = "gpt-4o-mini",
    ) -> tuple[bool, str, bool, str, str]: ...

    def sanitize_for_prompt(self, text: str) -> str: ...

    def redact_pii(self, text: str) -> Tuple[str, list[str]]: ...

    def check_prompt_injection(self, text: str) -> None: ...

    def moderate(self, text: str) -> None: ...


"""
class QuestionGenerator(Protocol):
    def next_question(
        self,
        *,
        mode: PracticeMode,
        role: str,
        seniority: str,
        jd: JobDescription | None,
        history: List[Message],
    ) -> str: ...
"""
