"""Fast smoke tests: `python -m pytest` or `python tests/test_smoke.py`."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.models.elo import expected_score
from src.models.match_model import match_probs
from src.models.svi_surface import SurvivalSurface, enforce_monotone
from src.markets.implied import implied_from_book
from src.markets.edges import kelly_fraction
import numpy as np


def test_match_probs_sum_to_one():
    p = match_probs(1900, 1700)
    assert abs(sum(p.values()) - 1.0) < 1e-9
    assert p["home"] > p["away"]  # favourite more likely


def test_survival_is_monotone():
    s = SurvivalSurface("X").fit({"group": 0.9, "QF": 0.3, "champion": 0.1}).survival()
    vals = list(s.values())
    assert all(vals[i] >= vals[i + 1] - 1e-9 for i in range(len(vals) - 1))


def test_enforce_monotone():
    out = enforce_monotone(np.array([0.5, 0.9, 0.2]))
    assert list(out) == [0.5, 0.5, 0.2]


def test_devig_sums_to_one():
    p = implied_from_book({"a": 0.6, "b": 0.55})
    assert abs(sum(p.values()) - 1.0) < 1e-9


def test_kelly_bounds():
    assert kelly_fraction(0.5, 0.5) == 0.0      # no edge -> no bet
    assert 0 <= kelly_fraction(0.6, 0.4) <= 0.25

def test_forward_simulation_returns_valid_shape():
    from app.streamlit_app import run_forward_simulation
    from src.models.tournament import GROUPS_2026
    import config

    all_teams = [t for g in GROUPS_2026.values() for t in g]
    elo = {team: config.ELO_BASE for team in all_teams}

    results = run_forward_simulation(
        tuple(sorted(elo.items())), 0.3131, 318.2, 5,
    )

    assert len(results) == 5
    for r in results:
        assert set(r.keys()) == {"R32", "R16", "QF", "SF", "champion"}
        assert r["R32"] >= r["R16"] >= r["QF"] >= r["SF"]  # each round is a subset of the previous
        assert r["champion"] in r["SF"]

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"ok  {name}")
    print("all smoke tests passed")
