"""
Purpose: Score/critique a user answer and propose improvements.
Powers the “Feedback” tab and learning loop.

What is inside (later):
score answer(answer, rubric, role, seniority, jd) -> Dict
(e.g., {star: 3/5, depth: 4/5, ...})
improve answer(answer, critique) -> str

Testing: Snapshot tests on known answers; bounds (empty/long answers).
"""

from __future__ import annotations

from typing import Any, Optional

from ..models import (
    HandoffMemo,
    InterviewerRole,
    InterviewerPersona,
    JobDescription,
    Message,
    LLMSettings,
    PracticeMode,
    InterviewStyle,
)
from ..interfaces import LLMClient, PromptFactory
from ..utils.llm_json import extract_json, require_object


def _clip_text(s: str, max_chars: int) -> str:
    """Clip text to max_chars, adding ellipsis if clipped."""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "…"


def _render_recent(history: list[Message], max_chars: int = 4000) -> str:
    """
    Render the recent (already-sliced) Q/A as a simple transcript.
    Controller should pass only the slice since the last memo.
    """
    lines: list[str] = []
    for m in history:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"Candidate: {content}")
        elif role == "assistant":
            lines.append(f"Interviewer: {content}")
    text = "\n\n".join(lines)
    return _clip_text(text, max_chars)


def _to_str_list(x: Any) -> list[str]:
    """Convert None, str, or list[str] to list[str].
    Discard empty/whitespace-only strings.
    """
    if x is None:
        return []
    if isinstance(x, list):
        out = []
        for item in x:
            if isinstance(item, str):
                item = item.strip()
                if item:
                    out.append(item)
        return out
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    return []


def _extract_last_qa(history: list[dict[str, str]]) -> tuple[str, str]:
    """
    Return (previous_assistant_question, last_user_answer) from the history.
    If not present, returns ("", "").
    """
    last_user = ""
    prev_assistant = ""

    for idx in range(len(history) - 1, -1, -1):
        m = history[idx]
        if m.get("role") == "user":
            last_user = (m.get("content") or "").strip()
            for j in range(idx - 1, -1, -1):
                if history[j].get("role") == "assistant":
                    prev_assistant = (history[j].get("content") or "").strip()
                    break
            break

    return prev_assistant, last_user


def generate_handoff_llm(
    *,
    llm: LLMClient,
    prompts: PromptFactory,
    settings: LLMSettings,
    interviewer_role: InterviewerRole,
    interviewer_persona: InterviewerPersona,
    jd: JobDescription | None,
    history_slice: list[Message],
    persona: InterviewerPersona,
    style: InterviewStyle,
    role: str,
    seniority: str,
    mode: PracticeMode,
) -> tuple[HandoffMemo, dict]:
    """Return (HandoffMemo, meta) from the recent transcript slice."""
    if not history_slice:
        return HandoffMemo(
            interviewer_role=interviewer_role,
            interviewer_persona=interviewer_persona,
            summary="No candidate answers were provided in this interviewer round.",
            strengths=[],
            concerns=[],
            recommendations=["Ask at least one question before switching interviewer."],
        ), {"tokens_in": 0, "tokens_out": 0}

    transcript_block = _render_recent(history_slice)

    system_prompt = prompts.build_handoff_system(
        persona=persona,
        style=style,
        role=role,
        seniority=seniority,
        mode=mode,
        interviewer_role=interviewer_role,
        jd=jd,
        memos=None,
    )
    user_prompt = prompts.handoff_instruction(
        transcript=transcript_block,
        interviewer_role=interviewer_role,
        interviewer_persona=interviewer_persona,
    )

    text, meta = llm.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        settings=settings,
    )

    obj = require_object(
        text, err="LLM did not return a valid JSON object for the handoff memo."
    )
    summary = (obj.get("summary") or "").strip() or "Handoff summary unavailable."
    strengths = _to_str_list(obj.get("strengths"))
    concerns = _to_str_list(obj.get("concerns"))
    recs = _to_str_list(obj.get("recommendations"))

    memo = HandoffMemo(
        interviewer_role=interviewer_role,
        interviewer_persona=interviewer_persona,
        summary=summary,
        strengths=strengths,
        concerns=concerns,
        recommendations=recs,
    )
    return memo, meta


def score_last_answer_llm(
    *,
    llm: LLMClient,
    prompts: PromptFactory,
    settings: LLMSettings,
    history: list[Message],
    role: str,
    seniority: str,
    mode: PracticeMode,
    interviewer_role: InterviewerRole,
    persona: InterviewerPersona,
    style: InterviewStyle,
    job: Optional[JobDescription],
    memos: list,
) -> tuple[dict, dict]:
    """Return (score_dict, meta) for the last Q/A in history."""
    q, a = _extract_last_qa(history)
    if not a:
        raise ValueError("No candidate answer found to score.")

    system = prompts.build_feedback_system(
        persona=persona,
        style=style,
        role=role,
        seniority=seniority,
        mode=mode,
        interviewer_role=interviewer_role,
        jd=job,
        memos=memos,
    )
    user_msg = prompts.scoring_instruction(question=q, answer=a)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    score_settings = LLMSettings(
        model=settings.model,
        temperature=min(settings.temperature, 0.3),
        top_p=settings.top_p,
        max_tokens=max(240, settings.max_tokens or 240),
    )
    text, meta = llm.chat(messages, score_settings)

    obj = extract_json(text) if text else {}
    scores = obj.get("scores") if isinstance(obj, dict) else {}

    def _clip01(v):
        try:
            return max(0.0, min(1.0, float(v)))
        except Exception:
            return 0.0

    normalized = {
        "answer_quality": _clip01(scores.get("answer_quality", 0)),
        "star_structure": _clip01(scores.get("star_structure", 0)),
        "technical_depth": _clip01(scores.get("technical_depth", 0)),
        "communication": _clip01(scores.get("communication", 0)),
    }
    summary = (obj.get("summary") or "").strip() if isinstance(obj, dict) else ""
    return {"scores": normalized, "summary": summary}, meta


def improve_last_answer_llm(
    *,
    llm: LLMClient,
    prompts: PromptFactory,
    settings: LLMSettings,
    history: list[Message],
    role: str,
    seniority: str,
    mode: PracticeMode,
    interviewer_role: InterviewerRole,
    persona: InterviewerPersona,
    style: InterviewStyle,
    job: Optional[JobDescription],
    memos: list,
    rescore: bool = False,
) -> tuple[dict, dict]:
    """Return (improvement_dict, meta) for the last Q/A in history."""
    q, a = _extract_last_qa(history)
    if not a:
        raise ValueError("No candidate answer found to improve.")

    system = prompts.build_feedback_system(
        persona=persona,
        style=style,
        role=role,
        seniority=seniority,
        mode=mode,
        interviewer_role=interviewer_role,
        jd=job,
        memos=memos,
    )
    user_msg = prompts.improve_answer_instruction(question=q, answer=a)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    improve_settings = LLMSettings(
        model=settings.model,
        temperature=min(settings.temperature, 0.5),
        top_p=settings.top_p,
        max_tokens=max(380, settings.max_tokens or 380),
    )
    text, meta = llm.chat(messages, improve_settings)
    obj = extract_json(text) if text else {}

    improved_answer = (
        (obj.get("improved_answer") or "").strip() if isinstance(obj, dict) else ""
    )
    next_actions = (
        (obj.get("next_actions") or "").strip() if isinstance(obj, dict) else ""
    )

    result = {"improved_answer": improved_answer, "next_actions": next_actions}

    if rescore and improved_answer:
        user_msg2 = prompts.scoring_instruction(question=q, answer=improved_answer)
        text2, meta2 = llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg2},
            ],
            LLMSettings(
                model=settings.model,
                temperature=0.2,
                top_p=settings.top_p,
                max_tokens=240,
            ),
        )
        obj2 = extract_json(text2) if text2 else {}
        scores2 = obj2.get("scores") if isinstance(obj2, dict) else None
        if isinstance(scores2, dict):
            for k, v in list(scores2.items()):
                try:
                    scores2[k] = max(0.0, min(1.0, float(v)))
                except Exception:
                    scores2[k] = 0.0
            result["scores"] = scores2
        if isinstance(meta, dict) and isinstance(meta2, dict):
            meta["tokens_in"] = int(meta.get("tokens_in", 0)) + int(
                meta2.get("tokens_in", 0)
            )
            meta["tokens_out"] = int(meta.get("tokens_out", 0)) + int(
                meta2.get("tokens_out", 0)
            )

    return result, meta
