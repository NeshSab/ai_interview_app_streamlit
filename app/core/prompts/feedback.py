"""Feedback/coaching prompts: scoring, improvement, handoff."""

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
)
from .common import jd_hint_block


def build_feedback_system(
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
    handoff_lines = ""
    if memos:
        bullets = "\n".join(
            f"- {m.interviewer_role.value} ("
            f"{getattr(m, 'interviewer_persona', '') if not hasattr(m, 'interviewer_persona') else m.interviewer_persona.value}"
            f"): {m.summary}"
            for m in memos
            if getattr(m, "summary", "")
        )
        if bullets:
            handoff_lines = f"\nPrevious handoff notes:\n{bullets}"

    return (
        "You are an interview feedback assistant and rubric grader.\n"
        "Use the session context to evaluate and coach the candidate.\n\n"
        f"- Target role: {role}\n"
        f"- Seniority: {seniority}\n"
        f"- Interviewer role: {interviewer_role.value}\n"
        f"- Mode: {mode.value}\n"
        f"- Desired tone: {persona.value} / {style.value}\n"
        f"{jd_hint_block(jd, max_chars=1000)}\n"
        f"{handoff_lines}\n\n"
        "Rules:\n"
        "- Be objective and concise.\n"
        "- Never invent facts absent from the candidate's answer.\n"
        "- When asked to return JSON, return EXACTLY one JSON object and nothing else."
    )


def scoring_instruction(*, question: str, answer: str) -> str:
    return dedent(
        f"""\
        Score the candidate's answer to the question using a 0..1 rubric.

        Question:
        {question or "(not available)"}

        Candidate answer:
        {answer or "(not available)"}

        Output ONLY this JSON object (no code fences, no commentary):
        {{
          "scores": {{
            "answer_quality":   <float 0..1>,
            "star_structure":   <float 0..1>,
            "technical_depth":  <float 0..1>,
            "communication":    <float 0..1>
          }},
          "summary": "<<= 80 words of concise feedback>"
        }}
        Notes:
        - If the mode is Behavioral, weight STAR more and technical less.
        - If the mode is Technical, weight technical_depth more.
        - Clamp each score to [0,1].
        """
    )


def improve_answer_instruction(*, question: str, answer: str) -> str:
    return dedent(
        f"""\
        Improve the candidate's answer to the question below while keeping it truthful
        to the original content (do not fabricate specifics like company names
         or numbers).
        Use concise, clear language; prefer STAR when relevant.

        Question:
        {question or "(not available)"}

        Original answer:
        {answer or "(not available)"}

        Output ONLY this JSON object (no code fences, no commentary):
        {{
          "improved_answer": "<= 180 words, concise, structured (STAR when relevant)>",
          "next_actions": "- Bullet 1\\n- Bullet 2\\n- Bullet 3"
        }}
        Constraints:
        - Keep details plausible and generic if the original is generic.
        - No headings or markdown beyond the hyphen bullets in next_actions.
        """
    )


def build_handoff_system(
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
    prior = ""
    if memos:
        lines = [
            (
                f"- {m.interviewer_role.value} ("
                f"{getattr(m, 'interviewer_persona', '') if not hasattr(m, 'interviewer_persona') else m.interviewer_persona.value}"
                f"): {m.summary}"
            )
            for m in memos
            if getattr(m, "summary", None)
        ]
        if lines:
            prior = "\nPrevious handoffs:\n" + "\n".join(lines)

    return (
        "You are an expert interviewer producing a concise handoff memo"
        " for the next interviewer.\n"
        f"Persona: {persona.value}\n"
        f"Style: {style.value}\n"
        f"Target role: {role}\n"
        f"Seniority: {seniority}\n"
        f"Mode: {mode.value}\n"
        f"Interviewer role: {interviewer_role.value}\n"
        f"{jd_hint_block(jd)}\n"
        f"{prior}\n"
        "Return a STRICT JSON object with keys exactly: summary, strengths, "
        "concerns, recommendations.\n"
        "No markdown, no code fences, no extra keys. If the interviewee"
        " did not provide reasonable response, say so."
    )


def handoff_instruction(
    *,
    transcript: str,
    interviewer_role: InterviewerRole,
    interviewer_persona: InterviewerPersona,
) -> str:
    return (
        f"Interviewer role: {interviewer_role.value}\n"
        f"Interviewer persona: {interviewer_persona.value}\n"
        "RECENT TRANSCRIPT (candidate=Candidate, interviewer=Interviewer):\n"
        f"{transcript}\n\n"
        "Return ONLY JSON like:\n"
        "{\n"
        '  "summary": "2-4 sentences, impact-first",\n'
        '  "strengths": ["...", "..."],\n'
        '  "concerns": ["...", "..."],\n'
        '  "recommendations": ["...", "..."]\n'
        "}"
    )
