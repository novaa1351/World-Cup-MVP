"""Convert market prices to implied probabilities and remove the vig.

Polymarket: a YES share trades in [0,1] and pays $1 on resolution, so the price
IS already an implied probability (minus a small spread). Multi-outcome books
still over-round, so we normalise.

Kalshi: prices are in cents (0-100); divide by 100 to get the implied prob.
"""
from __future__ import annotations
import numpy as np


def cents_to_prob(price_cents: float) -> float:
    return max(0.0, min(1.0, price_cents / 100.0))


def devig_proportional(raw_probs: dict[str, float]) -> dict[str, float]:
    """Proportional de-vig: scale a set of outcome probs so they sum to 1.

    `raw_probs` maps outcome -> raw implied prob (each already in [0,1]).
    The sum over a complete market is > 1 by the overround (the vig).
    """
    total = sum(raw_probs.values())
    if total <= 0:
        return {k: 0.0 for k in raw_probs}
    return {k: v / total for k, v in raw_probs.items()}


def overround(raw_probs: dict[str, float]) -> float:
    """Bookmaker overround (vig); ~0 for an efficient, liquid market."""
    return sum(raw_probs.values()) - 1.0


def implied_from_book(yes_prices: dict[str, float], in_cents: bool = False) -> dict[str, float]:
    """One call: take per-outcome YES prices -> de-vigged probabilities."""
    raw = {
        k: (cents_to_prob(v) if in_cents else max(0.0, min(1.0, v)))
        for k, v in yes_prices.items()
    }
    return devig_proportional(raw)


if __name__ == "__main__":
    book = {"Brazil": 0.20, "France": 0.18, "Argentina": 0.16, "field": 0.52}
    print("overround:", round(overround(book), 3))
    print(implied_from_book(book))
