"""
Purpose: Generate the next interview question(s) given role/seniority/history.
Why: Decouple question logic from the core chat. Enables banks, rotation,
adaptive difficulty.

What is inside (later):
next_question(history, role, seniority, jd, difficulty) -> str | List[str]

Strategies: behavioral, technical, system design, case studies.

Testing: Deterministic seeds for reproducible sequences; rules per role.
"""
