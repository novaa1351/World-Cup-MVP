"""Simulated paper-trading forward test.

Rules:
  - $1,000 starting bankroll (WCQ_PAPER_BANKROLL, default 1000.0).
  - Fractional-Kelly sizing capped at 25% of current bankroll per bet.
  - Only per-match W/D/L markets — tournament-level (champion/survival) markets
    are excluded because their settlement horizon is months out, making bankroll
    accounting and out-of-sample integrity much harder to verify.
  - Correlated-bet guard: no open bet on the same TEAM may exist when a new bet
    is considered. This matters because Kelly sizing assumes independent bets —
    stacking multiple bets on one team (e.g., Brazil to win their group-stage
    game AND their R32 game) violates that assumption. Both bets gain and lose
    together (correlated by Brazil's underlying strength), which inflates risk
    of ruin well beyond what Kelly predicts for independent events.
  - True out-of-sample: never reference an outcome before it happens. The
    correlated-bet check looks only at open (unsettled) bets, not results.

Configure:
  WCQ_PAPER_BANKROLL       — initial bankroll (default 1000.0)
  WCQ_KELLY_EDGE_THRESHOLD — minimum edge to place a bet (default 0.03)
"""
from __future__ import annotations
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

INITIAL_BANKROLL: float = float(os.environ.get("WCQ_PAPER_BANKROLL", "1000.0"))
KELLY_EDGE_THRESHOLD: float = float(os.environ.get("WCQ_KELLY_EDGE_THRESHOLD", "0.03"))
KELLY_CAP: float = 0.25  # hard cap — never bet more than 25% of bankroll


def _kelly(model_prob: float, market_prob: float) -> float:
    """Fractional Kelly stake, capped at KELLY_CAP, floored at 0."""
    from src.markets.edges import kelly_fraction
    return kelly_fraction(model_prob, market_prob, cap=KELLY_CAP)


def consider_bet(
    match_id: str,
    home_team: str,
    away_team: str,
    outcome: str,
    model_prob: float,
    market_prob: float,
    platform: str,
) -> dict | None:
    """Evaluate a potential paper bet. Places it if criteria are met.

    Args:
        match_id:    Stable identifier for the fixture (from storage.make_match_id).
        home_team:   Home side name (FIFA spelling).
        away_team:   Away side name.
        outcome:     "home" | "draw" | "away".
        model_prob:  Model's probability for this outcome.
        market_prob: Market's implied probability (de-vigged).
        platform:    "polymarket" | "kalshi".

    Returns a bet record dict if a bet was placed, None otherwise.
    """
    from src.bot.storage import (
        get_open_bets_for_team,
        get_current_bankroll,
        place_paper_bet,
    )
    from src.bot.notify import post_paper_bet_placed

    edge = model_prob - market_prob
    if edge < KELLY_EDGE_THRESHOLD:
        return None

    if market_prob <= 0 or market_prob >= 1:
        return None

    # Identify which team we'd be backing (for correlated-bet check)
    if outcome == "home":
        team_backed = home_team
    elif outcome == "away":
        team_backed = away_team
    else:
        team_backed = "Draw"

    # Correlated-bet guard — skip draws too (a draw bet is correlated with
    # both teams' defensive records, so we skip when either team has an open bet).
    teams_to_check = (
        [team_backed] if outcome not in ("draw",) else [home_team, away_team]
    )
    for team in teams_to_check:
        if get_open_bets_for_team(team):
            return None  # correlated position already open; skip

    bankroll = get_current_bankroll(INITIAL_BANKROLL)
    if bankroll <= 0:
        return None

    kf = _kelly(model_prob, market_prob)
    if kf <= 0:
        return None

    stake = kf * bankroll

    bet_id = place_paper_bet(
        match_id, home_team, away_team, team_backed, outcome, platform,
        market_prob, model_prob, kf, stake, bankroll,
    )

    post_paper_bet_placed(
        home_team, away_team, outcome, stake, kf, market_prob, model_prob,
        bankroll - stake,
    )

    return {
        "bet_id": bet_id,
        "match_id": match_id,
        "home_team": home_team,
        "away_team": away_team,
        "team_backed": team_backed,
        "outcome": outcome,
        "platform": platform,
        "market_prob": market_prob,
        "model_prob": model_prob,
        "edge": edge,
        "kelly_fraction": kf,
        "stake": stake,
        "bankroll_before": bankroll,
    }


def settle_match(match_id: str, winner: str) -> list[dict]:
    """Settle all open bets for a match. Returns list of settled bet records.

    winner: "home" | "draw" | "away"
    """
    from src.bot.storage import get_open_bets, settle_paper_bet

    all_open = get_open_bets()
    match_bets = [b for b in all_open if b["match_id"] == match_id]

    settled: list[dict] = []
    for bet in match_bets:
        result = settle_paper_bet(bet["id"], winner)
        if result:
            settled.append(result)

    return settled


def consider_bets_for_match(
    match_id: str,
    home_team: str,
    away_team: str,
    model_probs: dict[str, float],
    market_probs: dict[str, float],
    platform: str,
) -> list[dict]:
    """Evaluate all three outcomes for a fixture and place valid paper bets.

    In a real W/D/L market, you can buy one side; we evaluate each outcome
    independently and place at most one bet per match (the highest-edge outcome
    that clears the threshold, to avoid taking both sides of the same game).
    """
    candidates: list[tuple[float, str]] = []
    for outcome in ("home", "draw", "away"):
        mp = model_probs.get(outcome, 0.0)
        mk = market_probs.get(outcome, 0.0)
        if mk > 0:
            edge = mp - mk
            if edge >= KELLY_EDGE_THRESHOLD:
                candidates.append((edge, outcome))

    if not candidates:
        return []

    # Place only the single highest-edge bet per match (one bet = one exposure)
    candidates.sort(reverse=True)
    _, best_outcome = candidates[0]

    bet = consider_bet(
        match_id, home_team, away_team, best_outcome,
        model_probs[best_outcome], market_probs[best_outcome], platform,
    )
    return [bet] if bet else []


def post_pnl_update() -> None:
    """Post the current paper-trading P&L to Discord."""
    from src.bot.storage import get_pnl_summary
    from src.bot.notify import post_paper_pnl_report
    post_paper_pnl_report(get_pnl_summary())


if __name__ == "__main__":
    import tempfile, os as _os
    from src.bot.storage import init_db

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp = f.name
    init_db(tmp)

    # Simulate a bet placement
    _os.environ["DRY_RUN"] = "1"

    bet = consider_bet(
        "brazil_vs_morocco_2026-06-15",
        "Brazil", "Morocco", "home",
        model_prob=0.58, market_prob=0.50, platform="polymarket",
    )
    if bet:
        print(f"Placed bet: {bet['outcome']} stake=${bet['stake']:.2f} kelly={bet['kelly_fraction']:.2%}")
    else:
        print("No bet placed (threshold not met or correlation guard triggered)")

    _os.unlink(tmp)
