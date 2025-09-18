"""Facade that preserves your existing DefaultPromptFactory API."""

from __future__ import annotations
from typing import Optional, List
from core.models import (
    JobDescription,
    Message,
    PracticeMode,
    HandoffMemo,
    InterviewerRole,
    InterviewStyle,
    InterviewerPersona,
)
from . import interview as _interview
from . import feedback as _feedback
from . import job_description as _jd
from .common import assemble as _assemble


class DefaultPromptFactory:
    # INTERVIEW
    def build_system(
        self,
        *,
        persona: InterviewerPersona,
        style: InterviewStyle,
        role: str,
        seniority: str,
        mode: PracticeMode,
        interviewer_role: InterviewerRole,
        jd: Optional[JobDescription],
        memos: Optional[List[HandoffMemo]] = None,
    ) -> str:
        return _interview.build_interview_system(
            persona=persona,
            style=style,
            role=role,
            seniority=seniority,
            mode=mode,
            interviewer_role=interviewer_role,
            jd=jd,
            memos=memos,
        )

    def greeting_instruction(self) -> str:
        return _interview.greeting_instruction()

    # FEEDBACK / COACHING
    def build_feedback_system(
        self,
        *,
        persona: InterviewerPersona,
        style: InterviewStyle,
        role: str,
        seniority: str,
        mode: PracticeMode,
        interviewer_role: InterviewerRole,
        jd: Optional[JobDescription],
        memos: Optional[List[HandoffMemo]] = None,
    ) -> str:
        return _feedback.build_feedback_system(
            persona=persona,
            style=style,
            role=role,
            seniority=seniority,
            mode=mode,
            interviewer_role=interviewer_role,
            jd=jd,
            memos=memos,
        )

    def scoring_instruction(self, *, question: str, answer: str) -> str:
        return _feedback.scoring_instruction(question=question, answer=answer)

    def improve_answer_instruction(self, *, question: str, answer: str) -> str:
        return _feedback.improve_answer_instruction(question=question, answer=answer)

    def build_handoff_system(
        self,
        *,
        persona: InterviewerPersona,
        style: InterviewStyle,
        role: str,
        seniority: str,
        mode: PracticeMode,
        interviewer_role: InterviewerRole,
        jd: Optional[JobDescription],
        memos: Optional[List[HandoffMemo]] = None,
    ) -> str:
        return _feedback.build_handoff_system(
            persona=persona,
            style=style,
            role=role,
            seniority=seniority,
            mode=mode,
            interviewer_role=interviewer_role,
            jd=jd,
            memos=memos,
        )

    def handoff_instruction(
        self,
        *,
        transcript: str,
        interviewer_role: InterviewerRole,
        interviewer_persona: InterviewerPersona,
    ) -> str:
        return _feedback.handoff_instruction(
            transcript=transcript,
            interviewer_role=interviewer_role,
            interviewer_persona=interviewer_persona,
        )

    # JD PARSING
    def build_jd_parse_system(self) -> str:
        return _jd.build_jd_parse_system()

    def jd_parse_instruction(self, *, text: str) -> str:
        return _jd.jd_parse_instruction(text=text)

    def jd_requirements_only_instruction(self, *, role: str, seniority: str) -> str:
        return _jd.jd_requirements_only_instruction(role=role, seniority=seniority)

    def build_role_validation_system(self) -> str:
        return _jd.build_role_validation_system()

    def validate_role_seniority_instruction(
        self, *, role_text: str, seniority_text: str
    ) -> str:
        return _jd.validate_role_seniority_instruction(
            role_text=role_text, seniority_text=seniority_text
        )

    def build_plan_system(self) -> str:
        return _jd.build_plan_system()

    def jd_plan_with_requirements_instruction(
        self, *, role: str, seniority: str, requirements: list[str]
    ) -> str:
        return _jd.jd_plan_with_requirements_instruction(
            role=role, seniority=seniority, requirements=requirements
        )

    def assemble(
        self, *, system: str, history: list[Message], user_text: str
    ) -> list[Message]:
        return _assemble(system=system, history=history, user_text=user_text)
