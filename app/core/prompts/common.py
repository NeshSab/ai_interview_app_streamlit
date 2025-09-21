"""Shared prompt helpers used across prompt modules."""

from __future__ import annotations
from textwrap import dedent
from typing import Optional, List

from ..models import (
    PracticeMode,
    HandoffMemo,
    InterviewerRole,
    JobDescription,
    Message,
)


def mode_rules(mode: PracticeMode) -> str:
    if mode == PracticeMode.BEHAVIORAL:
        return (
            "Behavioral interview.\n"
            "- Focus on real past experiences (STAR).\n"
            "- Probe impact, metrics, and candidate's decisions.\n"
        )
    if mode == PracticeMode.TECHNICAL:
        return (
            "Technical interview.\n"
            "- Ask role-appropriate questions (concepts, design, trade-offs).\n"
            "- No preambles; one question at a time.\n"
        )
    return (
        "Situational interview.\n"
        "- Use realistic 'what would you do' scenarios.\n"
        "- Probe prioritization, stakeholders, risks.\n"
    )


def role_rules(role: InterviewerRole) -> str:
    if role == InterviewerRole.HR:
        return (
            "Interviewer role: HR/Recruiter.\n"
            "- Focus on motivation, communication, culture add, teamwork, "
            "and salary expectations.\n"
            "- No technical drills; keep questions accessible.\n"
        )
    if role == InterviewerRole.PM:
        return (
            "Interviewer role: PM/Stakeholder.\n"
            "- Focus on business impact, prioritization, trade-offs, storytelling.\n"
            "- Probe stakeholder alignment and decision rationale.\n"
        )
    return (
        "Interviewer role: Hiring Manager/Team Lead.\n"
        "- Mix technical depth and situational scenarios.\n"
        "- Probe execution details, quality, and ownership.\n"
    )


_FEWSHOT_BY_LEVEL = {
    PracticeMode.BEHAVIORAL: {
        "junior": [
            "Tell me about a time you asked for help effectively to unblock work. What did you try first?",  # noqa: E501
        ],
        "mid": [
            "Describe owning a deliverable end-to-end with measurable outcomes. How did you handle trade-offs?",  # noqa: E501
        ],
        "senior": [
            "Tell me about influencing a cross-team decision without authority. How did you win alignment?",  # noqa: E501
        ],
        "team_lead": [
            "Describe guiding a team through ambiguity. How did you de-risk and communicate across functions?",  # noqa: E501
        ],
    },
    PracticeMode.TECHNICAL: {
        "junior": [
            "Walk me through your approach before coding. How do you validate understanding and constraints?",  # noqa: E501
        ],
        "mid": [
            "What trade-offs and complexities would you call out in your design? How do you test the riskiest part?",  # noqa: E501
        ],
        "senior": [
            "How would you add observability (metrics/logs/alerts) and SLAs to your design?",  # noqa: E501
        ],
        "team_lead": [
            "How would you stage rollout, de-risk, and align stakeholders across teams?",
        ],
    },
    PracticeMode.SITUATIONAL: {
        "junior": [
            "When blocked by conflicting asks, how would you escalate and seek guidance appropriately?",  # noqa: E501
        ],
        "mid": [
            "How would you re-scope and communicate a plan when constraints tighten suddenly?",  # noqa: E501
        ],
        "senior": [
            "How do you align stakeholders under pressure and manage risk to business impact?",  # noqa: E501
        ],
        "team_lead": [
            "How would you renegotiate success criteria and communicate trade-offs to execs?",  # noqa: E501
        ],
    },
}


def few_shot_block(*, seniority: str, mode: PracticeMode, max_examples: int = 3) -> str:
    overlay = _FEWSHOT_BY_LEVEL.get(mode, {}).get(seniority, [])
    combined: list[str] = []
    for q in overlay:
        if q not in combined:
            combined.append(q)
        if len(combined) >= max_examples:
            break
    if not combined:
        return ""
    header = (
        f"Few-shot question exemplars for {seniority} ({mode.value}). "
        f"Do not copy verbatim:"
    )
    lines = [header] + [f"- {q}" for q in combined]
    return "\n" + "\n".join(lines)


def deliberate_block() -> str:
    return dedent(
        """\
        Deliberate reasoning (private):
        - Think step-by-step in a private scratchpad to
         choose the best single question next.
        - DO NOT include your scratchpad or reasoning in the output;
         output ONLY the question text.
        """
    )


def checklist_block() -> str:
    return dedent(
        """\
        Checklist to satisfy for each question:
        - Exactly one specific question.
        - ≤ 40 words unless deeper detail was explicitly requested.
        - Anchored in the role/JD or prior answers/handoffs.
        - Avoid compound/multi-part asks; one axis per turn.
        """
    )


def self_critique_block() -> str:
    return dedent(
        """\
        Self-critique (internal):
        - Draft 2 candidate questions privately.
        - Evaluate them for specificity, relevance, and word count.
        - Output ONLY the best single question; do not reveal drafts or critique.
        """
    )


def render_memos(memos: Optional[List[HandoffMemo]]) -> str:
    if not memos:
        return ""
    bullets = []
    for m in memos:
        role = getattr(m.interviewer_role, "value", m.interviewer_role)
        persona = getattr(m, "interviewer_persona", "")
        if hasattr(persona, "value"):
            persona = persona.value
        summary = getattr(m, "summary", "")
        bullets.append(f"- {role} ({persona}): {summary}")
    if not bullets:
        return ""
    return "\nPrevious interviewer handoffs:\n" + "\n".join(bullets)


def jd_hint_block(jd: Optional[JobDescription], *, max_chars: int = 1200) -> str:
    if jd and (jd.description or "").strip():
        return f"\nJob Description:\n{(jd.description or '')[:max_chars]}"
    return "Job Description: not provided."


def assemble(*, system: str, history: list[Message], user_text: str) -> list[Message]:
    return [
        {"role": "system", "content": system},
        *history,
        {"role": "user", "content": user_text},
    ]
