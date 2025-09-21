"""Prompts for generating interview prep roadmap infographic."""


def build_infographic_prompt(plan: dict) -> str:
    overview = plan.get("overview", {}) or {}
    role = overview.get("role", "Candidate")
    seniority = overview.get("seniority", "").title() or ""
    title = f"Roadmap of {seniority} {role} Interview Prep".strip()

    bullets = plan.get("summary_plan", []) or []
    if not bullets and plan.get("days"):
        bullets = [
            f"Day {i+1}: {d.get('title', 'Focus')}"
            for i, d in enumerate(plan["days"][:7])
        ]

    prep_steps = "\n".join(f"{i+1}. {b}" for i, b in enumerate(bullets))
    print(prep_steps)
    return f"""
    Design a clean, minimalist infographic poster titled "{title}".
    Description for each day:
    {prep_steps}
    Use exact description for each day as given above.
    Day 8 is the interview day, don't forget to wish a "Good luck!".

    Canvas size: 1024x1024 pixels (do not exceed; no cropping).
    Add 40px margin on all sides.
    Font sizes: poster title 20-22pt, headline for a step 14-16pt, description 10-12pt,
    icon size ~50x50 pixels. If needed reduce size to fit all days.
    Layout: below poster title, divide canvas into into 8 equal horizontal 
    rows stacked top-to-bottom: rows 1 to 7 = Day 1..Day 7,
    row 8 leave empty.

    Typography & sizing:
    - Besides title, each row should have: left = small flat icon;
    next to it = headline (“Day N”);
    below headline description.

    Styling:
    - Pastel palette, modern flat design, high readability, ample whitespace.
    - Consistent icon style. No watermarks or logos.    
    """
