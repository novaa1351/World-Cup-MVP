"""Post rich Discord embeds via webhook.

Config (all from environment — never hardcoded):
  DISCORD_WEBHOOK_URL        — fallback webhook used when a channel-specific var is unset
  DISCORD_WEBHOOK_DIGEST     — #daily-digest channel
  DISCORD_WEBHOOK_PRE_MATCH  — #pre-match channel
  DISCORD_WEBHOOK_POST_MATCH — #post-match channel
  DISCORD_WEBHOOK_PAPER      — #paper-trading channel
  DISCORD_WEBHOOK_LIVE       — #live-alerts channel
  DRY_RUN                    — set to "1" to print payloads instead of posting

If only DISCORD_WEBHOOK_URL is set, all notifications go to that one channel.
Set the channel-specific vars to route each type to its own channel.
"""
from __future__ import annotations
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import requests

WEBHOOK_URL: str = os.environ.get("DISCORD_WEBHOOK_URL", "")
DRY_RUN: bool = os.environ.get("DRY_RUN", "0").strip().lower() in ("1", "true", "yes")

# Per-channel webhooks — each falls back to WEBHOOK_URL if unset
_WH_DIGEST     = os.environ.get("DISCORD_WEBHOOK_DIGEST",     WEBHOOK_URL)
_WH_PRE_MATCH  = os.environ.get("DISCORD_WEBHOOK_PRE_MATCH",  WEBHOOK_URL)
_WH_POST_MATCH = os.environ.get("DISCORD_WEBHOOK_POST_MATCH", WEBHOOK_URL)
_WH_PAPER      = os.environ.get("DISCORD_WEBHOOK_PAPER",      WEBHOOK_URL)
_WH_LIVE       = os.environ.get("DISCORD_WEBHOOK_LIVE",       WEBHOOK_URL)

# Per-channel role IDs — set to ping a role when each notification posts.
# Get IDs from Discord: Server Settings → Roles → right-click role → Copy Role ID
# (requires Developer Mode: User Settings → Advanced → Developer Mode)
_ROLE_DIGEST     = os.environ.get("DISCORD_ROLE_DIGEST",     "")
_ROLE_PRE_MATCH  = os.environ.get("DISCORD_ROLE_PRE_MATCH",  "")
_ROLE_POST_MATCH = os.environ.get("DISCORD_ROLE_POST_MATCH", "")
_ROLE_PAPER      = os.environ.get("DISCORD_ROLE_PAPER",      "")
_ROLE_LIVE       = os.environ.get("DISCORD_ROLE_LIVE",       "")

# Maps each webhook URL to its role ID so _post() can add the mention automatically
_WEBHOOK_ROLE: dict[str, str] = {
    _WH_DIGEST:     _ROLE_DIGEST,
    _WH_PRE_MATCH:  _ROLE_PRE_MATCH,
    _WH_POST_MATCH: _ROLE_POST_MATCH,
    _WH_PAPER:      _ROLE_PAPER,
    _WH_LIVE:       _ROLE_LIVE,
}

# Discord embed color palette
_COL_GREEN  = 0x2ECC71
_COL_YELLOW = 0xF1C40F
_COL_RED    = 0xE74C3C
_COL_BLUE   = 0x3498DB
_COL_GREY   = 0x95A5A6

_FOOTER_DISCLAIMER = "Educational analysis — not betting advice. Model: Elo + calibrated draw."


def _post(
    payload: dict[str, Any],
    webhook_url: str = WEBHOOK_URL,
    dry_run: bool = DRY_RUN,
) -> None:
    """POST a Discord payload to the webhook, or print it in dry-run mode."""
    role_id = _WEBHOOK_ROLE.get(webhook_url, "")
    if role_id:
        payload = dict(payload)
        payload["content"] = f"<@&{role_id}>"
        payload["allowed_mentions"] = {"parse": [], "roles": [role_id]}
    if dry_run or not webhook_url:
        print("[DRY-RUN] Discord payload:")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    r = requests.post(webhook_url, json=payload, timeout=10)
    r.raise_for_status()


def _embed(
    title: str,
    description: str,
    color: int,
    fields: list[dict[str, Any]] | None = None,
    footer: str | None = None,
) -> dict[str, Any]:
    e: dict[str, Any] = {
        "title": title,
        "description": description,
        "color": color,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if fields:
        e["fields"] = fields
    if footer:
        e["footer"] = {"text": footer}
    return e


def _prob_bar(p: float, width: int = 10) -> str:
    """Compact ASCII bar for a probability in [0,1]."""
    filled = round(p * width)
    return "█" * filled + "░" * (width - filled)


def _fmt_edge(edge: float) -> str:
    sign = "+" if edge >= 0 else ""
    return f"{sign}{edge * 100:.1f}pp"


# ---------------------------------------------------------------------------
# Daily digest
# ---------------------------------------------------------------------------

def post_daily_digest(
    fixtures: list[dict],
    top_edges: list[dict],
    webhook_url: str = _WH_DIGEST,
    dry_run: bool = DRY_RUN,
) -> None:
    """Post today's fixture list with model W/D/L vs market probabilities.

    fixtures: list of dicts with keys home_team, away_team, kickoff_utc,
              kickoff_local (optional), model_probs (optional), market_probs (optional).
    top_edges: list of dicts from edges.edge_table(), already sorted by |edge|.
    """
    if not fixtures and not top_edges:
        return

    date_str = datetime.now(timezone.utc).strftime("%d %b %Y")
    lines: list[str] = []

    for f in fixtures:
        home, away = f["home_team"], f["away_team"]
        ko = f.get("kickoff_local") or f.get("kickoff_utc", "TBD")
        mp = f.get("model_probs", {})
        mk = f.get("market_probs", {})

        line = f"**{home} vs {away}** · {ko} UTC"
        if mp:
            model_str = (
                f"`W {mp.get('home', 0):.0%}  D {mp.get('draw', 0):.0%}  L {mp.get('away', 0):.0%}`"
            )
            line += f"\nModel {model_str}"
        if mk:
            mkt_str = (
                f"`W {mk.get('home', 0):.0%}  D {mk.get('draw', 0):.0%}  L {mk.get('away', 0):.0%}`"
            )
            line += f"  Market {mkt_str}"
        lines.append(line)

    description = "\n\n".join(lines) if lines else "No fixtures scheduled today."

    fields: list[dict[str, Any]] = []
    if top_edges:
        edge_lines = []
        for e in top_edges[:5]:
            outcome = str(e.get("outcome", "?"))
            edge = float(e.get("edge", 0))
            mp_val = float(e.get("model_prob", 0))
            mk_val = float(e.get("market_prob", 0))
            edge_lines.append(
                f"**{outcome}**: model {mp_val:.0%} · market {mk_val:.0%} · edge {_fmt_edge(edge)}"
            )
        fields.append({
            "name": "Biggest edges today",
            "value": "\n".join(edge_lines),
            "inline": False,
        })

    embed = _embed(
        title=f"⚽ WC 2026 Daily Digest — {date_str}",
        description=description,
        color=_COL_YELLOW,
        fields=fields or None,
        footer=_FOOTER_DISCLAIMER,
    )
    _post({"embeds": [embed]}, webhook_url, dry_run)


# ---------------------------------------------------------------------------
# Pre-match briefing
# ---------------------------------------------------------------------------

def post_pre_match_briefing(
    home_team: str,
    away_team: str,
    kickoff_utc: str,
    model_probs: dict[str, float],
    market_probs: dict[str, float] | None,
    elo_home: float,
    elo_away: float,
    platform: str = "Polymarket",
    webhook_url: str = _WH_PRE_MATCH,
    dry_run: bool = DRY_RUN,
) -> None:
    """Briefing posted ~1 hour before kickoff.

    Reports model vs market probabilities, per-outcome edges, the single biggest
    divergence, and the two model features driving the call (Elo gap + draw component).
    """
    labels = {"home": home_team, "draw": "Draw", "away": away_team}
    elo_gap = elo_home - elo_away

    prob_lines: list[str] = []
    for outcome in ("home", "draw", "away"):
        mp = model_probs.get(outcome, 0.0)
        mk = (market_probs or {}).get(outcome)
        bar = _prob_bar(mp)
        row = f"**{labels[outcome]}**: {bar} {mp:.0%}"
        if mk is not None:
            edge = mp - mk
            row += f"  ·  Market {mk:.0%}  ·  {_fmt_edge(edge)}"
        prob_lines.append(row)

    # Feature 1 — Elo gap and what it implies
    if elo_gap > 50:
        fav, underdog = home_team, away_team
        strength = "heavy" if abs(elo_gap) > 200 else ("clear" if abs(elo_gap) > 100 else "slight")
    elif elo_gap < -50:
        fav, underdog = away_team, home_team
        strength = "heavy" if abs(elo_gap) > 200 else ("clear" if abs(elo_gap) > 100 else "slight")
    else:
        fav = underdog = None
        strength = "evenly matched"

    feature_1 = (
        f"Elo: **{home_team}** {elo_home:.0f}  vs  **{away_team}** {elo_away:.0f}"
        f"  (gap {elo_gap:+.0f})"
    )
    feature_2 = (
        f"{'**' + fav + '** is the ' + strength + ' favourite' if fav else 'Teams are **evenly matched**'}"
    )
    # Feature 2 — draw component
    draw_p = model_probs.get("draw", 0.0)
    feature_3 = f"P(draw) = {draw_p:.0%} (calibrated exponential-decay draw model)"

    # Biggest divergence
    biggest_line = ""
    if market_probs:
        divergences = [
            (abs(model_probs.get(o, 0) - market_probs.get(o, 0)), o)
            for o in ("home", "draw", "away")
        ]
        divergences.sort(reverse=True)
        _, big_outcome = divergences[0]
        big_edge = model_probs.get(big_outcome, 0) - market_probs.get(big_outcome, 0)
        direction = "above" if big_edge > 0 else "below"
        biggest_line = (
            f"Biggest divergence: **{labels[big_outcome]}** — model {_fmt_edge(big_edge)} {direction} {platform}"
        )

    fields: list[dict[str, Any]] = [
        {"name": "Probabilities (model · market · edge)", "value": "\n".join(prob_lines), "inline": False},
        {"name": "Model drivers", "value": "\n".join([feature_1, feature_2, feature_3]), "inline": False},
    ]
    if biggest_line:
        fields.append({"name": "Key signal", "value": biggest_line, "inline": False})
    if not market_probs:
        fields.append({
            "name": "Market data",
            "value": f"No per-match {platform} market found — showing model only.",
            "inline": False,
        })

    embed = _embed(
        title=f"📋 Pre-match: {home_team} vs {away_team}",
        description=f"Kickoff: **{kickoff_utc[:16]} UTC**",
        color=_COL_BLUE,
        fields=fields,
        footer=_FOOTER_DISCLAIMER,
    )
    _post({"embeds": [embed]}, webhook_url, dry_run)


# ---------------------------------------------------------------------------
# Post-match scorecard
# ---------------------------------------------------------------------------

def post_post_match_scorecard(
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
    model_probs: dict[str, float],
    market_probs: dict[str, float] | None,
    calibration: dict,
    webhook_url: str = _WH_POST_MATCH,
    dry_run: bool = DRY_RUN,
) -> None:
    """Scorecard posted after full time. Calibration scoreboard is the centrepiece."""
    if home_score > away_score:
        winner = "home"
    elif home_score == away_score:
        winner = "draw"
    else:
        winner = "away"

    labels = {"home": home_team, "draw": "Draw", "away": away_team}
    result_str = f"**{home_team} {home_score}–{away_score} {away_team}**"

    model_p = model_probs.get(winner, 0.0)
    market_p = (market_probs or {}).get(winner)

    prob_line = f"Model gave **{labels[winner]}**: {model_p:.0%}"
    if market_p is not None:
        prob_line += f" · Market: {market_p:.0%}"
        if model_p >= market_p:
            verdict = "✅ Model was better calibrated on this match"
        else:
            verdict = "❌ Market was better calibrated on this match"
        prob_line += f"\n{verdict}"

    # Calibration scoreboard — be candid
    calib_lines: list[str] = []
    n = calibration.get("n_matches", 0)
    bm = calibration.get("brier_model")
    bmk = calibration.get("brier_market")
    lm = calibration.get("logloss_model")
    lmk = calibration.get("logloss_market")

    if bm is not None and bmk is not None:
        brier_model_winning = bm < bmk  # lower Brier = better
        brier_emoji = "📈" if brier_model_winning else "📉"
        calib_lines.append(
            f"{brier_emoji} Brier: Model **{bm:.4f}** · Market **{bmk:.4f}**"
            + (" ← model winning" if brier_model_winning else " ← market winning")
        )
    elif bm is not None:
        calib_lines.append(f"Brier (model): {bm:.4f} (no market baseline yet)")

    if lm is not None and lmk is not None:
        ll_model_winning = lm < lmk
        ll_emoji = "📈" if ll_model_winning else "📉"
        calib_lines.append(
            f"{ll_emoji} Log-loss: Model **{lm:.4f}** · Market **{lmk:.4f}**"
            + (" ← model winning" if ll_model_winning else " ← market winning")
        )

    if n:
        calib_lines.append(f"Sample: {n} match{'es' if n != 1 else ''}")

    if not calib_lines:
        calib_lines = ["Not enough data yet — calibration updates after each settled match."]

    # Colour the embed by whether model beat market on this match
    if market_p is not None:
        color = _COL_GREEN if model_p >= market_p else _COL_RED
    else:
        color = _COL_GREY

    fields: list[dict[str, Any]] = [
        {"name": "Result", "value": f"{result_str}\n{prob_line}", "inline": False},
        {"name": "Running calibration scoreboard", "value": "\n".join(calib_lines), "inline": False},
    ]

    embed = _embed(
        title=f"📊 Final: {home_team} vs {away_team}",
        description=result_str,
        color=color,
        fields=fields,
    )
    _post({"embeds": [embed]}, webhook_url, dry_run)


# ---------------------------------------------------------------------------
# Live in-play alert
# ---------------------------------------------------------------------------

def post_live_edge_alert(
    home_team: str,
    away_team: str,
    outcome: str,
    model_prob: float,
    market_prob: float,
    edge: float,
    platform: str,
    trigger: str,
    webhook_url: str = _WH_LIVE,
    dry_run: bool = DRY_RUN,
) -> None:
    """Alert for a live edge crossing threshold or a sharp market reprice."""
    labels = {"home": home_team, "draw": "Draw", "away": away_team}
    outcome_label = labels.get(outcome, outcome)
    color = _COL_GREEN if edge > 0 else _COL_RED

    embed = _embed(
        title=f"⚡ Live: {home_team} vs {away_team}",
        description=(
            f"**{outcome_label}** · Model {model_prob:.0%} · {platform} {market_prob:.0%} "
            f"· edge {_fmt_edge(edge)}\n_{trigger}_"
        ),
        color=color,
        footer=_FOOTER_DISCLAIMER,
    )
    _post({"embeds": [embed]}, webhook_url, dry_run)


def post_cross_platform_spread(
    home_team: str,
    away_team: str,
    outcome: str,
    poly_prob: float,
    kalshi_prob: float,
    webhook_url: str = _WH_LIVE,
    dry_run: bool = DRY_RUN,
) -> None:
    """Alert when Polymarket and Kalshi disagree sharply on the same outcome."""
    labels = {"home": home_team, "draw": "Draw", "away": away_team}
    outcome_label = labels.get(outcome, outcome)
    spread = abs(poly_prob - kalshi_prob)
    cheaper = "Polymarket" if poly_prob < kalshi_prob else "Kalshi"

    embed = _embed(
        title=f"🔄 Cross-platform spread: {home_team} vs {away_team}",
        description=(
            f"**{outcome_label}** · Polymarket {poly_prob:.0%} · Kalshi {kalshi_prob:.0%}\n"
            f"Spread: {spread * 100:.1f}pp · Cheaper on {cheaper}"
        ),
        color=_COL_YELLOW,
        footer=_FOOTER_DISCLAIMER,
    )
    _post({"embeds": [embed]}, webhook_url, dry_run)


def post_paper_bet_placed(
    home_team: str,
    away_team: str,
    outcome: str,
    stake: float,
    kelly_fraction: float,
    market_prob: float,
    model_prob: float,
    bankroll: float,
    webhook_url: str = _WH_PAPER,
    dry_run: bool = DRY_RUN,
) -> None:
    """Notification when a paper bet is placed."""
    labels = {"home": home_team, "draw": "Draw", "away": away_team}
    outcome_label = labels.get(outcome, outcome)

    embed = _embed(
        title=f"💰 Paper bet: {home_team} vs {away_team}",
        description=(
            f"**{outcome_label}** · Stake ${stake:.2f} "
            f"({kelly_fraction:.0%} Kelly · {_fmt_edge(model_prob - market_prob)} edge)\n"
            f"Bankroll after stake: ${bankroll:.2f}"
        ),
        color=_COL_BLUE,
        footer="Paper trade only — no real money.",
    )
    _post({"embeds": [embed]}, webhook_url, dry_run)


def post_paper_pnl_report(
    summary: dict,
    webhook_url: str = _WH_PAPER,
    dry_run: bool = DRY_RUN,
) -> None:
    """P&L summary for the paper trading ledger."""
    bankroll = summary.get("current_bankroll", 1000.0)
    roi = summary.get("roi", 0.0)
    wagered = summary.get("total_wagered", 0.0)
    breakdown = summary.get("breakdown", {})

    won = breakdown.get("won", {})
    lost = breakdown.get("lost", {})
    open_ = breakdown.get("open", {})

    lines = [
        f"Bankroll: **${bankroll:.2f}** ({'+' if roi >= 0 else ''}{roi:.1%} ROI)",
        f"Total wagered: ${wagered:.2f}",
        f"Won: {won.get('n', 0)} bets (${won.get('pnl', 0):.2f})",
        f"Lost: {lost.get('n', 0)} bets (${lost.get('pnl', 0):.2f})",
        f"Open: {open_.get('n', 0)} bets",
    ]

    color = _COL_GREEN if roi >= 0 else _COL_RED
    embed = _embed(
        title="📈 Paper trading P&L",
        description="\n".join(lines),
        color=color,
        footer="Out-of-sample forward test — results update after each settled match.",
    )
    _post({"embeds": [embed]}, webhook_url, dry_run)


if __name__ == "__main__":
    # Dry-run demo of every embed type
    os.environ["DRY_RUN"] = "1"
    from importlib import reload
    import src.bot.notify as _self
    reload(_self)

    print("=== Daily digest ===")
    post_daily_digest(
        fixtures=[{
            "home_team": "Brazil", "away_team": "Morocco",
            "kickoff_utc": "2026-06-15T18:00:00Z",
            "model_probs": {"home": 0.55, "draw": 0.24, "away": 0.21},
            "market_probs": {"home": 0.52, "draw": 0.26, "away": 0.22},
        }],
        top_edges=[{"outcome": "Brazil", "model_prob": 0.55, "market_prob": 0.52, "edge": 0.03, "edge_pct": 3.0}],
        dry_run=True,
    )

    print("\n=== Pre-match briefing ===")
    post_pre_match_briefing(
        "Brazil", "Morocco", "2026-06-15T18:00:00Z",
        {"home": 0.55, "draw": 0.24, "away": 0.21},
        {"home": 0.52, "draw": 0.26, "away": 0.22},
        1820.0, 1680.0, dry_run=True,
    )

    print("\n=== Post-match scorecard ===")
    post_post_match_scorecard(
        "Brazil", "Morocco", 2, 0,
        {"home": 0.55, "draw": 0.24, "away": 0.21},
        {"home": 0.52, "draw": 0.26, "away": 0.22},
        {"n_matches": 3, "brier_model": 0.210, "brier_market": 0.225,
         "logloss_model": 0.680, "logloss_market": 0.720},
        dry_run=True,
    )

    print("\n=== Live edge alert ===")
    post_live_edge_alert("Brazil", "Morocco", "home", 0.62, 0.50, 0.12, "Polymarket",
                         "Model edge crossed 10pp threshold", dry_run=True)
