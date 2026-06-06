"""Central config: paths and API endpoints. No secrets here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
for _p in (RAW, PROCESSED):
    _p.mkdir(parents=True, exist_ok=True)

# --- Data sources -----------------------------------------------------------
# martj42 keeps the Kaggle "international results 1872->" dataset mirrored on
# GitHub, so we can pull it with no Kaggle auth.
HISTORICAL_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

# --- Prediction-market endpoints (public, read-only, no API key) ------------
POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"   # /markets, /events
KALSHI_BASE = "https://external-api.kalshi.com/trade-api/v2"  # /markets, /events, /series/{t}

# --- Modeling knobs ---------------------------------------------------------
ELO_K = 40              # Elo update speed for WC-level matches
ELO_HOME_ADV = 65       # Elo points added to the non-neutral host
ELO_BASE = 1500         # rating for a team we have never seen
MC_SIMS = 20_000        # Monte-Carlo tournament simulations

# Tournament "round depth" axis used by the SVI-style survival surface.
# Order matters: deeper round => smaller survival probability (no-arb monotone).
ROUNDS = ["group", "R32", "R16", "QF", "SF", "final", "champion"]
