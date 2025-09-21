"""
Purpose: The single orchestration point for a session. Owns state, history, job.
It centralizes “one-turn” logic and session lifecycle (start, chat_once, reset).
Prevents UI from knowing how prompts/LLM/services work.

Key responsibilities:
- Hold and sync history (list of messages) and job (JD).
- Build a system prompt using prompts.factory.
- Assemble messages (system + history + new user input).
- Call services.llm_openai (via LLMClient interface).
- Apply guardrails (services.security).
- Return (reply_text, meta) where meta contains token usage, etc.
- reset() clears internal state (history, job, token counters).

Testing: Pure unit tests with fakes: mock LLMClient, PromptFactory,
SecurityGuard. Verify message assembly and error handling.
"""

from __future__ import annotations
from typing import Optional, Mapping, Any

from .models import (
    PracticeMode,
    SessionState,
    HandoffMemo,
    Message,
    JobDescription,
    LLMSettings,
    InterviewerRole,
    InterviewerPersona,
    InterviewStyle,
)
from .services import jd_analyzer
from .interfaces import LLMClient, PromptFactory, SecurityGuard
from .prompts.factory import DefaultPromptFactory
from .services.security import DefaultSecurity
from .services.answer_critic import (
    generate_handoff_llm,
    score_last_answer_llm,
    improve_last_answer_llm,
)
from .services.pricing import estimate_tokens_from_text
from .services.speech import tts_bytes
from .services.voice import transcribe_wav_bytes


class InterviewSessionController:
    def __init__(
        self,
        llm: LLMClient,
    ):
        self.llm: LLMClient = llm
        self.prompts: PromptFactory = DefaultPromptFactory()
        self.security: SecurityGuard = DefaultSecurity()
        self.state = SessionState()

        self.tokens_in: int = 0
        self.tokens_out: int = 0
        self.model_used: Optional[str] = None

    def is_ready(self) -> bool:
        """True if the controller is ready to chat (has an LLM)."""
        return self.llm is not None

    def set_interviewer(
        self, role: InterviewerRole, persona: InterviewerPersona
    ) -> None:
        """Set the committed interviewer role/persona for the session."""
        self.state.interviewer_role = role
        self.state.interviewer_persona = persona

    def set_history(self, messages: list[Message]) -> None:
        """Overwrite the full history with a new list of messages."""
        self.state.history = messages[:]

    def get_history(self) -> list[Message]:
        """Get the current full history of messages."""
        return self.state.history

    def append_assistant(self, text: str) -> None:
        """Append an assistant message to history."""
        self.state.history.append({"role": "assistant", "content": text})

    def append_user(self, text: str) -> None:
        """Append a user message to history."""
        self.state.history.append({"role": "user", "content": text})

    def set_job_description(self, jd_obj) -> None:
        """Set or clear the current JobDescription for the session."""

        if jd_obj is None:
            self.state.job = None
            return

        if isinstance(jd_obj, JobDescription):
            self.state.job = jd_obj
            return

        if isinstance(jd_obj, dict):
            data = dict(jd_obj)
        else:
            data = getattr(jd_obj, "__dict__", None)
            if not isinstance(data, dict):
                raise TypeError(f"Unsupported JD type: {type(jd_obj)!r}")

        data.setdefault("meta", data.get("meta") or {})
        self.state.job = JobDescription(**data)

    def reset(self, *, preserve_job: bool = True) -> None:
        """Clear history, handoffs, cursors, and token counters. Optionally keep job."""
        job = self.state.job if preserve_job else None
        self.state = SessionState()
        self.state.job = job
        self.tokens_in = self.tokens_out = 0
        self.model_used = None

    def generate_handoff(
        self,
        *,
        settings: Optional[LLMSettings] = None,
        persona: "InterviewerPersona" = None,
        style: "InterviewStyle" = None,
        mode: PracticeMode = PracticeMode.BEHAVIORAL,
    ) -> HandoffMemo:
        """
        Slice recent history since last memo, ask LLM for a handoff memo,
        append it to state, and advance the cursor. Returns the memo.
        """
        persona = persona or self.state.interviewer_persona
        style = style or InterviewStyle.CONCISE
        use_settings = settings or LLMSettings(
            model="gpt-4o-mini", temperature=0.2, top_p=1.0, max_tokens=500
        )
        start = int(self.state.last_memo_cursor or 0)
        recent = self.state.history[start:] if self.state.history else []

        memo, meta = generate_handoff_llm(
            llm=self.llm,
            prompts=self.prompts,
            settings=use_settings,
            interviewer_role=self.state.interviewer_role,
            interviewer_persona=self.state.interviewer_persona,
            jd=self.state.job,
            history_slice=recent,
            persona=persona,
            style=style,
            role=(
                self.state.job.title
                if (self.state.job and self.state.job.title)
                else ""
            ),
            seniority="",
            mode=mode,
        )

        self.tokens_in += int(meta.get("tokens_in", 0))
        self.tokens_out += int(meta.get("tokens_out", 0))
        self.model_used = use_settings.model

        self.state.handoffs.append(memo)
        self.state.last_memo_cursor = len(self.state.history)

        return memo, meta

    def chat_once(
        self,
        *,
        settings: LLMSettings,
        role: str,
        persona: InterviewerPersona,
        style: InterviewStyle,
        seniority: str,
        mode: PracticeMode,
        interviewer_role: InterviewerRole,
        user_text: Optional[str] = None,
    ) -> tuple[str, dict]:
        """
        Handles a normal back-and-forth chat turn (user input + AI reply).
        Pattern:
        Takes the latest user input (either explicitly passed as user_text or
        pulled from history).
        Validates input (security guard).
        Builds system prompt and full history with that user message.
        Sends to LLM.
        Returns whatever the assistant says (could be answer, critique, feedback, etc.).
        Output: a general LLM reply and metadata.
        So chat_once is user-driven → candidate says something, interviewer responds.
        """
        if user_text is None:
            user_utt = next(
                (
                    m["content"]
                    for m in reversed(self.state.history)
                    if m.get("role") == "user"
                ),
                "",
            )
        else:
            user_utt = user_text

        self.security.validate_user_input(user_utt)
        self.security.moderate(user_utt)
        user_utt = self.security.sanitize_for_prompt(user_utt)
        user_utt, pii = self.security.redact_pii(user_utt)
        self.security.check_prompt_injection(user_utt)

        system_prompt = self.prompts.build_system(
            persona=persona,
            style=style,
            role=role,
            seniority=seniority,
            mode=mode,
            interviewer_role=interviewer_role,
            jd=self.state.job,
            memos=self.state.handoffs,
        )
        messages = self.prompts.assemble(
            system=system_prompt, history=self.state.history, user_text=user_utt
        )

        reply, meta = self.llm.chat(messages, settings)
        self.append_assistant(reply)

        self.tokens_in += int(meta.get("tokens_in", 0))
        self.tokens_out += int(meta.get("tokens_out", 0))
        self.model_used = settings.model
        return reply, meta

    def greet_and_open(
        self,
        *,
        persona: InterviewerPersona,
        style: InterviewStyle,
        interviewer_role: InterviewerRole,
        role: str,
        seniority: str,
        mode: PracticeMode,
        settings: LLMSettings,
    ) -> tuple[str, dict]:
        """
        One assistant turn: brief greeting + invite the candidate to introduce
        themselves.
        """
        system_prompt = self.prompts.build_system(
            persona=persona,
            style=style,
            role=role,
            seniority=seniority,
            mode=mode,
            interviewer_role=interviewer_role,
            jd=self.state.job,
            memos=self.state.handoffs,
        )
        greet_instr = self.prompts.greeting_instruction()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": greet_instr},
        ]

        reply, meta = self.llm.chat(messages, settings)
        self.append_assistant(reply)
        self.tokens_in += int(meta.get("tokens_in", 0))
        self.tokens_out += int(meta.get("tokens_out", 0))
        self.model_used = settings.model
        return reply, meta

    def score_last_answer(
        self,
        *,
        settings: LLMSettings,
        role: str,
        seniority: str,
        mode: PracticeMode,
        interviewer_role: InterviewerRole,
        persona: "InterviewerPersona",
        style: "InterviewStyle",
    ) -> tuple[dict, dict]:
        """
        Ask the LLM to score the last candidate answer with a 0..1 rubric.
        Returns (result_dict, meta). The result has keys:
        - scores: {answer_quality, star_structure, technical_depth, communication}
        - summary: short text feedback
        """
        result, meta = score_last_answer_llm(
            llm=self.llm,
            prompts=self.prompts,
            settings=settings,
            history=self.state.history,
            role=role,
            seniority=seniority,
            mode=mode,
            interviewer_role=interviewer_role,
            persona=persona,
            style=style,
            job=self.state.job,
            memos=self.state.handoffs,
        )
        self.tokens_in += int(meta.get("tokens_in", 0))
        self.tokens_out += int(meta.get("tokens_out", 0))
        self.model_used = settings.model
        return result, meta

    def improve_last_answer(
        self,
        *,
        settings: LLMSettings,
        role: str,
        seniority: str,
        mode: PracticeMode,
        interviewer_role: InterviewerRole,
        persona: "InterviewerPersona",
        style: "InterviewStyle",
        rescore: bool = False,
    ) -> tuple[dict, dict]:
        """
        Ask the LLM to rewrite the last candidate answer and propose next actions.
        Returns (result_dict, meta). The result has keys:
        - improved_answer
        - next_actions
        - (optional) scores  # if rescore=True
        """
        result, meta = improve_last_answer_llm(
            llm=self.llm,
            prompts=self.prompts,
            settings=settings,
            history=self.state.history,
            role=role,
            seniority=seniority,
            mode=mode,
            interviewer_role=interviewer_role,
            persona=persona,
            style=style,
            job=self.state.job,
            memos=self.state.handoffs,
            rescore=rescore,
        )
        self.tokens_in += int(meta.get("tokens_in", 0))
        self.tokens_out += int(meta.get("tokens_out", 0))
        self.model_used = settings.model
        return result, meta

    def process_text_input(self, text: str, *, model: str = "gpt-4o-mini"):
        """
        Convenience: returns (jd_dict, meta) suitable for JobDescription(**jd_dict)
        with controller-managed token accounting.
        """
        jd_dict, meta = jd_analyzer.process_text_input(text, self.llm, model=model)
        self.tokens_in += int(meta.get("tokens_in", 0))
        self.tokens_out += int(meta.get("tokens_out", 0))
        self.model_used = model
        return jd_dict, meta

    def process_pdf(self, file_like, *, model: str = "gpt-4o-mini"):
        """
        Convenience: returns (jd_dict, meta) suitable for JobDescription(**jd_dict)
        with controller-managed token accounting.
        """
        jd_dict, meta = jd_analyzer.process_pdf(file_like, self.llm, model=model)
        self.tokens_in += int(meta.get("tokens_in", 0))
        self.tokens_out += int(meta.get("tokens_out", 0))
        self.model_used = model
        return jd_dict, meta

    def generate_prep_plan(
        self,
        *,
        role: str,
        seniority: str,
        requirements: str,
        model: str = "gpt-4o-mini",
    ):
        """
        Generate plan separately (after role/seniority are known).
        """
        plan, meta = jd_analyzer.generate_prep_plan(
            role=role,
            seniority=seniority,
            requirements=requirements,
            llm=self.llm,
            model=model,
        )

        self.tokens_in += int(meta.get("tokens_in", 0))
        self.tokens_out += int(meta.get("tokens_out", 0))
        self.model_used = model
        return plan, meta

    def speak(self, text: str, *, voice="alloy", tts_model="gpt-4o-mini-tts"):
        """Returns audio bytes and meta with char/token counts from text."""
        safe = (text or "").strip()
        if not safe:
            return b"", {"tts_chars": 0, "tts_tokens_est": 0, "model": tts_model}
        audio = tts_bytes(safe, self.llm, voice=voice, model=tts_model)
        est = estimate_tokens_from_text(safe)

        self.tokens_in += est
        self.tokens_out += 0
        self.model_used = tts_model

        return audio, {
            "tts_chars": len(safe),
            "tts_tokens_est": est,
            "model": tts_model,
        }

    def voice_to_text(self, wav_bytes: bytes, *, model="whisper-1") -> str:
        """Returns (text, meta) from audio bytes."""
        text = transcribe_wav_bytes(wav_bytes, self.llm, model=model)
        est = estimate_tokens_from_text(text)

        self.tokens_in += 0
        self.tokens_out += est
        self.model_used = model

        return text, {
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "model": self.model_used,
        }
