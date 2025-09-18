from typing import Dict, List


def build_infographic_prompt(plan: Dict) -> str:
    overview = plan.get("overview", {}) or {}
    role = overview.get("role", "Candidate")
    seniority = overview.get("seniority", "").title() or ""
    title = f"7-Day {seniority} {role} Interview Prep Roadmap".strip()

    bullets: List[str] = plan.get("summary_plan", []) or []
    if not bullets and plan.get("days"):
        bullets = [
            f"Day {i+1}: {d.get('title', 'Focus')}"
            for i, d in enumerate(plan["days"][:7])
        ]

    steps = "\n".join(f"{i+1}. {b}" for i, b in enumerate(bullets[:7]))

    return f"""
        Design a clean, minimalist infographic roadmap poster titled "{title}".
        Layout: vertical 7-step timeline, numbered circles, icons, short headlines.
        Style: pastel colors, modern flat design, high readability.
        Content:
        {steps}
        """
