from __future__ import annotations
from typing import Optional, Any

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

from core.interfaces import LLMClient
from core.services.llm_openai import OpenAILLMClient
from core.models import LLMSettings, JDParseResult, ParseOutcome
from core.prompts.factory import DefaultPromptFactory
from core.services.security import DefaultSecurity
from core.utils.llm_json import extract_json, require_object

_MAX_REQ = 12
_MAX_REQ_CHARS = 1200


def extract_pdf_text(file_like) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(file_like)
        parts = []
        for page in reader.pages:
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            if txt.strip():
                parts.append(txt)
        return "\n\n".join(parts).strip()
    except Exception:
        return ""


def _sum_meta(*metas: dict | None) -> dict[str, int]:
    ti = sum(int(m.get("tokens_in", 0)) for m in metas if m)
    to = sum(int(m.get("tokens_out", 0)) for m in metas if m)
    return {"tokens_in": ti, "tokens_out": to}


def parse_job_text(
    text: str,
    *,
    llm: Optional[LLMClient] = None,
    model: str = "gpt-4o-mini",
) -> tuple[JDParseResult, dict[str, int]]:
    """
    LLM-driven parse: classify + extract (role, seniority, company, location,
    requirements).
    Returns: (JDParseResult, meta)
    """
    client = llm or OpenAILLMClient()
    prompts = DefaultPromptFactory()
    security = DefaultSecurity()

    sys = prompts.build_jd_parse_system()
    usr = prompts.jd_parse_instruction(text=text)

    settings = LLMSettings(model=model, temperature=0.3, top_p=0.9, max_tokens=900)
    out, meta_parse = client.chat(
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
        settings=settings,
    )
    obj = extract_json(out)

    classification = (obj.get("classification") or "").strip().lower()
    if classification not in {"full_jd", "role_seniority_only", "neither"}:
        classification = "neither"

    role_raw = (obj.get("role") or "").strip()
    seniority_raw = (obj.get("seniority") or "").strip()

    # Validate + normalize role/seniority
    ok_role, norm_role, ok_sen, norm_sen, notes_val, meta_val = (
        security.validate_role_and_seniority(role_raw, seniority_raw, model=model)
    )

    company = (obj.get("company") or "").strip() or None
    location = (obj.get("location") or "").strip() or None

    # Extract requirements if present
    reqs = obj.get("requirements") or []
    reqs = [str(r).strip() for r in reqs if str(r).strip()][:12]
    req_src = (obj.get("requirements_source") or "").strip().lower()
    if req_src not in {"specific", "general"}:
        req_src = "specific" if reqs else "general"

    # If we only had role/seniority (or JD missing specifics), add a second pass to
    # create *general* requirements ONLY (plan is separate now).
    meta_req = None
    if (classification == "role_seniority_only" or not reqs) and (
        ok_role and (ok_sen or norm_sen)
    ):
        sys2 = prompts.build_jd_parse_system()
        usr2 = prompts.jd_requirements_only_instruction(
            role=norm_role or role_raw,
            seniority=(norm_sen or seniority_raw) or "",
        )
        out2, meta_req = client.chat(
            messages=[
                {"role": "system", "content": sys2},
                {"role": "user", "content": usr2},
            ],
            settings=LLMSettings(
                model=model, temperature=0.3, top_p=1.0, max_tokens=450
            ),
        )
        obj2 = extract_json(out2)
        reqs2 = obj2.get("requirements") or []
        new_reqs = [str(r).strip() for r in reqs2 if str(r).strip()][:12]
        if new_reqs:
            reqs = new_reqs
            req_src = "general"

    try:
        confidence = float(obj.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
    except Exception:
        confidence = 0.0

    notes_all = (obj.get("notes") or "").strip()
    if notes_val and notes_val not in notes_all:
        notes_all = (notes_all + (" " if notes_all else "") + notes_val).strip()

    result = JDParseResult(
        outcome=ParseOutcome(classification),
        proposed_role=(norm_role or (role_raw or None)),
        proposed_seniority=(norm_sen or None),
        company=company,
        location=location,
        requirements=reqs,
        requirements_source="general" if req_src == "general" else "specific",
        confidence=confidence,
        raw_text=text,
        notes=notes_all,
    )
    return result, _sum_meta(meta_parse, meta_val, meta_req)


def _normalize_requirements(reqs: Any) -> list[str]:
    """
    Accepts a str (newline/comma separated) or list[str].
    Returns de-duped, trimmed list capped to _MAX_REQ, with a char budget.
    """
    items = []
    if isinstance(reqs, str):
        raw = reqs.replace(",", "\n")
        items = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    elif isinstance(reqs, list):
        items = [str(x).strip() for x in reqs if str(x).strip()]

    # de-dup in order
    seen = set()
    out = []
    char_budget = _MAX_REQ_CHARS
    for it in items:
        if it.lower() in seen:
            continue
        if char_budget - len(it) <= 0:
            break
        out.append(it)
        seen.add(it.lower())
        char_budget -= len(it)
        if len(out) >= _MAX_REQ:
            break
    return out


def generate_prep_plan(
    *,
    role: str,
    seniority: str,
    requirements: Any,  # str or list[str]
    llm: Optional[LLMClient] = None,
    model: str = "gpt-4o-mini",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Produce a 7-day plan JSON using role, seniority, and (optional) requirements.
    Returns (plan_dict, meta). The controller should consume `meta` for token accounting.
    """
    client = llm or OpenAILLMClient()
    prompts = DefaultPromptFactory()
    sec = DefaultSecurity()

    # Sanitize inputs a bit (won’t modify meaning)
    role_s = sec.sanitize_for_prompt(role or "")
    seniority_s = sec.sanitize_for_prompt(seniority or "")
    req_list = _normalize_requirements(requirements)
    req_list = [sec.sanitize_for_prompt(r) for r in req_list]

    system = prompts.build_plan_system()
    user = prompts.jd_plan_with_requirements_instruction(
        role=role_s, seniority=seniority_s, requirements=req_list
    )

    settings = LLMSettings(model=model, temperature=0.3, top_p=0.9, max_tokens=3000)
    text, meta = client.chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        settings=settings,
    )

    obj = require_object(text, err="LLM did not return a valid plan JSON.")
    obj.setdefault("overview", {})
    obj.setdefault("days", [])
    obj.setdefault("summary_plan", [])

    if not isinstance(obj["days"], list):
        obj["days"] = []
    if not isinstance(obj["summary_plan"], list):
        obj["summary_plan"] = []

    return obj, meta


def process_text_input(
    text: str, *, model: str = "gpt-4o-mini"
) -> tuple[dict, dict[str, int]]:
    """
    Return a dict compatible with core.models.JobDescription(**dict) + meta.
    Raises ValueError when text is 'neither'.
    """
    security = DefaultSecurity()
    text = security.sanitize_for_prompt(text)
    text, _ = security.redact_pii(text)
    text = security.validate_job_description_length(text)

    parsed, meta = parse_job_text(text, model=model)
    if parsed.outcome == ParseOutcome.NEITHER:
        raise ValueError(
            "This text doesn't look like a job description or a role/seniority line."
        )

    meta_dict: dict[str, object] = {
        "seniority": parsed.proposed_seniority or "",
        "source": "text",
        "parse_outcome": parsed.outcome.value,
        "confidence": parsed.confidence,
        "requirements": parsed.requirements,
        "requirements_source": parsed.requirements_source,  # "specific" | "general"
        # plan intentionally excluded here (separate step)
        "notes": parsed.notes,
    }
    if parsed.company:
        meta_dict["company_detected"] = parsed.company
    if parsed.location:
        meta_dict["location_detected"] = parsed.location

    jd_dict = {
        "title": parsed.proposed_role or "(role not detected)",
        "company": parsed.company,
        "location": parsed.location,
        "description": text.strip(),
        "meta": meta_dict,
    }
    return jd_dict, meta


def process_pdf(file_like, *, model: str = "gpt-4o-mini") -> Optional[dict]:
    pdf_text = extract_pdf_text(file_like)
    if not pdf_text:
        return None
    return process_text_input(pdf_text, model=model)
