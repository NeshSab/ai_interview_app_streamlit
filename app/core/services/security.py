"""
Purpose: Guardrails for inputs and content.
Content: early, predictable failures; prevent oversized requests, PII,
or prompt-injection from JDs.
"""

from ..utils.llm_json import extract_json
from ..models import LLMSettings
from ..prompts.factory import DefaultPromptFactory
from ..interfaces import LLMClient
import re

EMAIL = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
CCARD = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

INJECTION_CUES = [
    "ignore previous",
    "disregard previous",
    "system prompt",
    "as the assistant",
    "you must not follow",
    "new instructions:",
    "### system",
    "BEGIN SYSTEM",
]

BANNED_WORDS = {
    "profanity": ["fuck", "shit", "bitch"],
    "discrimination": ["retard", "tranny", "bimbo"],
    "violence": ["kill", "punch", "slap"],
    "sexual": ["sexy", "milf", "hot"],
}

MAX_INPUT_CHARS = 8000
MAX_JD_CHARS = 10000


class DefaultSecurity:
    def validate_user_input(self, text: str) -> None:
        if not text.strip():
            raise ValueError("Please enter a non-empty message.")
        if len(text) > MAX_INPUT_CHARS:
            raise ValueError("Re-type your answer.\nYour message is too long.")

    def validate_job_description_length(self, text: str) -> str:
        if len(text) > MAX_JD_CHARS:
            text = text[:MAX_JD_CHARS]
        return text

    def sanitize_for_prompt(self, text: str) -> str:
        return (text or "").replace("\x00", "").strip()

    def redact_pii(self, text: str):
        found = []

        def _redact(rx, label):
            nonlocal text, found
            if rx.search(text):
                found.append(label)
                text = rx.sub(f"[{label}]", text)

        _redact(EMAIL, "EMAIL")
        _redact(PHONE, "PHONE")
        _redact(CCARD, "CARD")
        _redact(SSN, "SSN")
        return text, found

    def check_prompt_injection(self, text: str) -> None:
        t = (text or "").lower()
        if any(k in t for k in INJECTION_CUES):
            raise ValueError(
                "Re-type your answer.\n"
                + "Suspected prompt-injection phrasing in the input."
            )

    def moderate(self, text: str) -> None:
        """
        Enhanced local check for banned words.
        Use regex to match whole words and return detailed violations.
        """
        text = (text or "").lower()
        violations = {}

        for category, words in BANNED_WORDS.items():
            pattern = r"\b(?:" + "|".join(re.escape(word) for word in words) + r")\b"
            matches = re.findall(pattern, text)
            if matches:
                violations[category] = matches

        if violations:
            raise ValueError(
                f"Re-type your answer.\n"
                f"Content violates usage guidelines: {violations}"
            )

    def validate_role_and_seniority(
        self,
        role_text: str,
        seniority_text: str,
        llm: LLMClient,
        *,
        model: str = "gpt-4o-mini",
    ) -> tuple[bool, str, bool, str, str, dict]:
        """
        Returns:
        (ok_role, normalized_role, ok_seniority, normalized_seniority, notes, meta)
        meta contains token usage from the LLM call.
        """

        if not (role_text or "").strip():
            return (
                False,
                "",
                False,
                "",
                "Role is empty.",
                {"tokens_in": 0, "tokens_out": 0},
            )

        client = llm
        prompts = DefaultPromptFactory()
        system = prompts.build_role_validation_system()
        user = prompts.validate_role_seniority_instruction(
            role_text=role_text, seniority_text=seniority_text
        )

        text, meta = client.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            settings=LLMSettings(
                model=model, temperature=0.3, top_p=0.9, max_tokens=220
            ),
        )
        obj = extract_json(text)

        ok_role = bool(obj.get("ok_role"))
        normalized_role = (obj.get("normalized_role") or "").strip()
        ok_seniority = bool(obj.get("ok_seniority"))
        normalized_seniority = (obj.get("normalized_seniority") or "").strip()
        notes = (obj.get("notes") or "").strip()

        if ok_role and not normalized_role:
            normalized_role = role_text.strip()

        return (
            ok_role,
            normalized_role,
            ok_seniority,
            normalized_seniority,
            notes,
            meta,
        )
