"""
Purpose: Token math & cost estimation.
Central pricing logic so UI/controller do not duplicate calculations.
"""

from ..models import Price


PRICE_TABLE = {
    "gpt-5-mini": Price(0.25, 2.00),
    "gpt-4o-mini": Price(0.15, 0.60),
    "gpt-4o": Price(2.50, 10.00),
    "gpt-4.1-mini": Price(0.40, 1.60),
}


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    p = PRICE_TABLE.get(model, Price(0.0, 0.0))
    return (tokens_in / 1000000) * p.input_per_1M + (
        tokens_out / 1000000
    ) * p.output_per_1M


def estimate_tokens_from_text(text: str) -> int:
    """Fast heuristic: ~4 chars per token."""
    t = (text or "").strip()
    if not t:
        return 0

    return (len(t) + 3) // 4
