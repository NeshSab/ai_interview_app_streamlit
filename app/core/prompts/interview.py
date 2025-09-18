"""Interview/chat prompts (greeting, next question)"""

from __future__ import annotations
from textwrap import dedent
from typing import Optional, List

from core.models import (
    JobDescription,
    PracticeMode,
    HandoffMemo,
    InterviewerRole,
    InterviewStyle,
    InterviewerPersona,
    PromptProfile,
)
from .common import (
    role_rules,
    mode_rules,
    few_shot_block,
    deliberate_block,
    checklist_block,
    self_critique_block,
    jd_hint_block,
)
ACTIVE_PROMPT_PROFILE: PromptProfile = PromptProfile.FEW_SHOT


def build_interview_system(
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
    core = dedent(
        f"""\
        You are the interviewer. {role_rules(interviewer_role)}.
        Stay in character. Your persona: {persona.value} and style: {style.value}.
        You interview for target role of {role} with seniority: {seniority}.
        Interview mode:
        {mode_rules(mode)}
        Rules:
        - Ask exactly ONE specific question at a time.
        - No preamble, no explanations, no bullet lists unless explicitly requested.
        - Keep it under 40 words unless deeper detail was requested.
        - If job description is provided, use it to inform your questions.
         {jd_hint_block(jd)}
        """
    )

    # attach previous handoffs (if any)
    if memos:
        from .common import render_memos

        core += "\n" + render_memos(memos)

    # attach profile technique
    technique = ""
    if ACTIVE_PROMPT_PROFILE == PromptProfile.ZERO_SHOT:
        pass
    elif ACTIVE_PROMPT_PROFILE == PromptProfile.FEW_SHOT:
        # NOTE: assumes `seniority` already normalized like "junior|mid|senior|team_lead"
        technique = few_shot_block(seniority=seniority, mode=mode)
    elif ACTIVE_PROMPT_PROFILE == PromptProfile.DELIBERATE:
        technique = "\n" + deliberate_block()
    elif ACTIVE_PROMPT_PROFILE == PromptProfile.CHECKLIST:
        technique = "\n" + checklist_block()
    elif ACTIVE_PROMPT_PROFILE == PromptProfile.SELF_CRITIQUE:
        technique = "\n" + self_critique_block()

    return core + (("\n" + technique) if technique else "")


def greeting_instruction() -> str:
    return (
        "Write the next message as the interviewer.\n"
        "- Use the context provided in the system prompt (interviewer role, persona,"
        " mode, JD) to set tone and scope.\n"
        "- In 2 to 3 sentences, greet the candidate and state who you are and "
        "what this round will focus on.\n"
        "- Then ask the candidate to briefly introduce themselves (1 to 2 sentences)"
        " relevant to the role.\n"
        "- Ask no other questions yet. Output a single short paragraph. No headings,"
        " no bullet points, no markdown."
    )
