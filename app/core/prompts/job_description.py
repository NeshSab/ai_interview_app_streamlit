"""Job Description parsing / validation prompts."""

from __future__ import annotations
from textwrap import dedent


def build_jd_parse_system() -> str:
    return (
        "You are a recruiting analyst. Your task is to classify and extract "
        "information from job postings, or detect when the text is not a job posting. "
        "You might see abbreviations such as 'Jr DS', or misspellings—if these are "
        "clear job-related terms, interpret them; if not, return empty.\n"
        "Be conservative and return only the JSON object requested."
    )


def jd_parse_instruction(*, text: str) -> str:
    return f"""
            Analyze the text below.

            First, set "classification" to one of:
            - "full_jd": reasonably complete job ad with responsibilities/requirements
            - "role_seniority_only": only role/title and seniority,
             little/no requirements
            - "neither": not a job posting

            Extract if available: role/title, seniority
             (e.g., Junior/Mid-level/Senior/Team Lead),
            company, location, and a list of concrete requirements.

            If requirements are missing but role+seniority exist,
             you may still propose a short preparation plan.

            Return ONLY this JSON:
            {{
            "classification": "full_jd" | "role_seniority_only" | "neither",
            "role": "<string or empty>",
            "seniority": "<string or empty>",
            "company": "<string or empty>",
            "location": "<string or empty>",
            "requirements": ["<strings>"],
            "requirements_source": "specific" | "general",
            "confidence": <float 0..1>,
            "notes": "<short note>"
            }}

            Text:
            \"\"\"{text.strip()}\"\"\"
            """.strip()


def jd_requirements_only_instruction(*, role: str, seniority: str) -> str:
    return f"""
    Given ROLE="{role}" and SENIORITY="{seniority}",
    list general requirements/skills for the position.

    Output ONLY this JSON:
    {{
        "requirements": ["<8 to 12 concise bullets>"]
    }}
    """.strip()


def build_role_validation_system() -> str:
    return (
        "You are a strict validator for job role and seniority inputs.\n"
        "Goal: confirm the role looks like a realistic job title and map seniority "
        'to one of: "junior", "mid", "senior", "team_lead".\n'
        "Return ONLY the JSON object requested—no commentary, no code fences."
    )


def validate_role_seniority_instruction(*, role_text: str, seniority_text: str) -> str:
    role_text = (role_text or "").strip()
    seniority_text = (seniority_text or "").strip()
    return (
        "Validate and normalize the following:\n"
        f"role='{role_text}'\n"
        f"seniority='{seniority_text}'\n\n"
        "Rules:\n"
        "- ok_role: true if the role is a plausible job title (e.g., 'Data Analyst',"
        "'Frontend Developer').\n"
        "- normalized_seniority: one of ['junior','mid','senior','team_lead'] "
        "if the input clearly maps; else empty.\n"
        "- ok_seniority: true only if a clear mapping exists.\n"
        "- normalized_role: a tidied version of the title (e.g., proper casing),"
        " or empty if not ok.\n"
        "Keep it conservative; if unsure, set ok_* = false and explain in notes.\n\n"
        "Output ONLY this JSON object:\n"
        "{\n"
        "  'ok_role': true|false,\n"
        "  'normalized_role': '<string or empty>',\n"
        "  'ok_seniority': true|false,\n"
        "  'normalized_seniority': '<junior|mid|senior|team_lead|>',\n"
        "  'notes': '<short reason or guidance>'\n"
        "}"
    )


def build_plan_system() -> str:
    """
    System prompt for producing a 7-day preparation plan.
    Output is a strict JSON object, easy to render in UI.
    """
    return dedent(
        """\
        You are an experienced technical interview coach.
        Create a 7-day preparation plan tailored to the target role and seniority.

        Return a STRICT JSON object with EXACTLY these keys:
        {
            "overview": {
            "role": "<string>",
            "seniority": "<string>",
            "daily_time_minutes": <int 60..240>,
            "focus_weights": {
                "fundamentals": <float 0..1>,
                "role_specific": <float 0..1>,
                "mock_review": <float 0..1>
            }
            },
            "days": [
            {
                "day": "Day 1",
                "theme": "<5-8 words>",
                "goals": ["<1-3 concise goals>"],
                "tasks": [
                {"label": "Warm-up", "minutes": <int>, "items": ["<1-3 short items>"]},
                {"label": "Deep work", "minutes": <int>, "items": ["<1-3 short items>"]},
                {"label": "Applied practice", "minutes": <int>, "items": ["<1-3 short items>"]},
                {"label": "Cool-down", "minutes": <int>, "items": ["<1-2 short items>"]}
                ],
                "deliverables": ["<0-3 concrete outputs>"]
            }
            // ... Days 2..7 in the same structure ...
            ],
            "summary_plan": [
            "Day 1: <one-line>",
            "Day 2: <one-line>",
            "Day 3: <one-line>",
            "Day 4: <one-line>",
            "Day 5: <one-line>",
            "Day 6: <one-line>",
            "Day 7: <one-line>"
            ]
        }

        HARD CONSTRAINTS:
        - Output EXACTLY one JSON object. No markdown, no code fences, no commentary.
        - Provide EXACTLY 7 entries in "days" (Day 1 through Day 7).
        - Each day's total "minutes" across tasks should sum to ~90 to 80.
        - Keep every text item concise (≤ 18 words) and action-oriented.
        - Avoid guessing company names or revealing private data.
        - If role/seniority are generic or unclear, produce a sensible general plan
        and keep it labeled via the theme wording.

        STYLING:
        - Clear, skimmable, bullet-ready phrasing.
        - Prefer verbs at the start of items (e.g., "Review", "Implement", "Practice").
        """
    )


def jd_plan_with_requirements_instruction(
    *, role: str, seniority: str, requirements: list[str]
) -> str:
    role_s = (role or "").strip()
    seniority_s = (seniority or "").strip()

    req_lines = "\n".join(f"- {r}" for r in requirements) if requirements else "(none)"

    return dedent(
        f"""\
            Create a 7-day interview preparation plan tailored to:
            - ROLE: "{role_s or "(unspecified)"}"
            - SENIORITY: "{seniority_s or "(unspecified)"}"

            Use these ROLE-SPECIFIC REQUIREMENTS as primary guidance:
            {req_lines}

            Follow the JSON schema and constraints from the system prompt.
            Return ONLY the JSON object (no prose, no code fences).
            """
    )
