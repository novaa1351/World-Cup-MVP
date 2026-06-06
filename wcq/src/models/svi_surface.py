"""SVI-style survival surface for tournament forecasting.

WHAT TRANSFERS FROM OPTIONS SVI (and what doesn't) -- read this before the demo:

In options, raw SVI fits total implied variance across log-moneyness `k`:

    w(k) = a + b * ( rho*(k - m) + sqrt((k - m)^2 + sigma^2) )

It is valued for three properties, and *those* are what we reuse here:
  1. A smooth, low-parameter curve fit to sparse, noisy market quotes.
  2. Hard no-arbitrage constraints baked into the fit (butterfly => valid
     risk-neutral density; calendar => total variance monotone in maturity).
  3. Least-squares calibration to whatever quotes the market gives us.

The cross-domain mapping we make explicit:
  options strike/moneyness axis  ->  team latent-strength axis
  options maturity axis          ->  tournament ROUND DEPTH axis
  implied-vol surface value      ->  P(team survives to that round)
  butterfly no-arb (valid pdf)   ->  per-round survival probs form a valid
                                     distribution over teams (sum constraints)
  calendar no-arb (monotone w)   ->  survival is MONOTONE NON-INCREASING in
                                     round depth: P(champion) <= P(final) <=
                                     ... <= P(advance from group)

HONEST CAVEAT (say this in interviews -- it is a strength, not a weakness):
the literal SVI hyperbola is tuned to vol smiles; there is no a-priori reason
it is the *correct* shape for a survival curve. So what we genuinely borrow is
the METHODOLOGY (smooth + no-arb-constrained + market-calibrated interpolation),
not the exact equation. The statistically grounded engine for the raw numbers
is the Monte-Carlo bracket sim (tournament.py); this surface is the calibration
and reconciliation layer that snaps the model onto sparse market anchors while
preserving no-arbitrage structure.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.optimize import minimize

import config

ROUNDS = config.ROUNDS
DEPTH = {r: i for i, r in enumerate(ROUNDS)}  # 0=group ... 6=champion


def raw_svi(k: np.ndarray, a, b, rho, m, sigma) -> np.ndarray:
    """The literal options SVI total-variance function (kept for reference /
    for fitting the strength-axis smile when you have a row of quotes)."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def enforce_monotone(curve: np.ndarray) -> np.ndarray:
    """Project a survival curve onto the no-arb monotone-non-increasing set.

    This is the calendar-no-arbitrage analogue: you cannot be likelier to win
    the cup than to reach the final. We clip each round to <= the previous one
    and into [0, 1].
    """
    out = np.clip(curve.astype(float), 0.0, 1.0)
    for i in range(1, len(out)):
        out[i] = min(out[i], out[i - 1])
    return out


class SurvivalSurface:
    """Per-team survival curve over the round-depth axis, with an SVI-style
    smooth + monotone fit that can be calibrated to sparse market anchors.

    Parameterisation along depth d in {0..6}: a smooth decay
        S(d) = sigmoid( alpha - beta * d )
    (2 params/team, like a mini-SVI). `alpha` encodes team strength (higher =>
    survives longer); `beta` encodes how fast its odds decay round to round.
    We fit (alpha, beta) by least squares to anchor points -- those anchors can
    be the Monte-Carlo survival probs, the market-implied probs, or a blend --
    then hard-enforce monotonicity so the output is always arbitrage-free.
    """

    def __init__(self, team: str):
        self.team = team
        self.alpha = 0.0
        self.beta = 1.0

    @staticmethod
    def _curve(alpha: float, beta: float) -> np.ndarray:
        d = np.arange(len(ROUNDS))
        return 1.0 / (1.0 + np.exp(-(alpha - beta * d)))

    def fit(self, anchors: dict[str, float], weights: dict[str, float] | None = None):
        """Calibrate to anchors, e.g. {'group': 0.86, 'champion': 0.07}.

        anchors maps round name -> target survival probability.
        """
        names = list(anchors)
        d = np.array([DEPTH[r] for r in names], dtype=float)
        y = np.array([anchors[r] for r in names], dtype=float)
        w = np.array([(weights or {}).get(r, 1.0) for r in names], dtype=float)

        def loss(theta):
            a, b = theta
            pred = 1.0 / (1.0 + np.exp(-(a - b * d)))
            return float(np.sum(w * (pred - y) ** 2))

        res = minimize(loss, x0=[2.0, 0.6], method="Nelder-Mead")
        self.alpha, self.beta = float(res.x[0]), float(res.x[1])
        return self

    def survival(self) -> dict[str, float]:
        """No-arb survival probability at every round."""
        curve = enforce_monotone(self._curve(self.alpha, self.beta))
        return {r: float(curve[DEPTH[r]]) for r in ROUNDS}


def build_surface(team_survivals: dict[str, dict[str, float]]) -> np.ndarray:
    """Stack many teams' survival curves into a (teams x rounds) matrix for the
    heatmap/3D 'surface' chart. Also useful for the butterfly-style check that
    each round's column is a sensible distribution across teams."""
    return np.array([[team_survivals[t][r] for r in ROUNDS] for t in team_survivals])


if __name__ == "__main__":
    s = SurvivalSurface("Brazil").fit({"group": 0.88, "QF": 0.34, "champion": 0.12})
    for rnd, p in s.survival().items():
        print(f"  {rnd:9s} {p:5.1%}")
