"""
Canonical data shapes, shared truth for typing/validation between layers.

Typical contents:
- LLMSettings (model, temperature, top_p, max_tokens).
- JobDescription (title, company, location, description, meta).
- Message (role, content).

Testing: Trivial; mostly types. Add validation helpers if needed.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum
from datetime import datetime


class PracticeMode(str, Enum):
    BEHAVIORAL = "Behavioral"
    TECHNICAL = "Technical"
    SITUATIONAL = "Situational"


class InterviewerRole(str, Enum):
    HR = "HR / Recruiter"
    HIRING_MANAGER = "Hiring Manager / Team Lead"
    PM = "PM / Stakeholder"


class InterviewerPersona(str, Enum):
    FRIENDLY = "Friendly"
    NEUTRAL = "Neutral"
    STRICT = "Strict"


class InterviewStyle(str, Enum):
    CONCISE = "Concise"
    DETAILED = "Detailed"


class ParseOutcome(str, Enum):
    FULL_JD = "full_jd"
    ROLE_SENIORITY_ONLY = "role_seniority_only"
    NEITHER = "neither"


class PromptProfile(str, Enum):
    ZERO_SHOT = "Zero-shot"
    FEW_SHOT = "Few-shot (Q exemplars)"
    DELIBERATE = "Chain-of-Thought (private)"
    CHECKLIST = "Checklist constraints"
    SELF_CRITIQUE = "Self-critique (refine)"


@dataclass
class JDParseResult:
    outcome: ParseOutcome
    proposed_role: Optional[str]
    proposed_seniority: Optional[str]
    company: Optional[str]
    location: Optional[str]
    requirements: list[str]
    requirements_source: Literal["specific", "general"]
    confidence: float
    raw_text: str
    notes: str = ""


@dataclass
class Message:
    role: str
    content: str


@dataclass
class HandoffMemo:
    interviewer_role: InterviewerRole
    interviewer_persona: InterviewerPersona
    summary: str
    strengths: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    start_idx: int = 0
    end_idx: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SessionState:
    job: Optional[JobDescription] = None

    history: list[Message] = field(default_factory=list)

    interviewer_role: InterviewerRole = InterviewerRole.HR
    interviewer_persona: InterviewerPersona = InterviewerPersona.NEUTRAL

    handoffs: list[HandoffMemo] = field(default_factory=list)
    last_memo_cursor: int = 0


@dataclass
class JobDescription:
    title: str
    company: Optional[str]
    location: Optional[str]
    description: str
    meta: dict


@dataclass
class LLMSettings:
    model: str
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 512
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    response_format: Optional[dict] = None


@dataclass(frozen=True)
class Price:
    input_per_1M: float
    output_per_1M: float
