"""In-play 1X2 win-probability model for regulation-time soccer markets.

Goals for each side are modelled as independent Poisson processes. The two
full-match scoring rates are backed out from the pre-match market prices, so
the model inherits the market's view of team strength; it only adds the
mechanics of *how much an event should move the price*. That makes it usable
for any match without per-team training data.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.stats import poisson

REGULATION_MINUTES = 90.0
# Expected added time per half; matters late in games when every minute counts.
STOPPAGE_MINUTES = {1: 3.0, 2: 6.0}
EFFECTIVE_MINUTES = REGULATION_MINUTES + sum(STOPPAGE_MINUTES.values())
# Empirical scoring-rate multipliers after a red card (e.g. Vecer et al. 2009,
# Titman et al. 2015): the short-handed side scores ~1/3 less, opponents ~1/4 more.
RED_CARD_OWN_FACTOR = 0.67
RED_CARD_OPP_FACTOR = 1.25
MAX_GOALS = 12
EXTRA_TIME_MINUTES = 30.0
# Prior for matches priced only by a two-way "to advance" market.
TYPICAL_TOTAL_GOALS = 2.5


@dataclass(frozen=True)
class MatchState:
    minute: float = 0.0
    home_goals: int = 0
    away_goals: int = 0
    home_reds: int = 0
    away_reds: int = 0

    def apply(self, event: str, team: str | None) -> "MatchState":
        """Return the state after a match event ("goal", "red_card", "goal_disallowed")."""
        side = {"home": "home", "away": "away"}.get(team or "")
        if event in ("goal_disallowed", "red_card", "goal") and side is None:
            raise ValueError(f"{event} needs team='home' or 'away'")
        if event == "goal":
            return replace(self, **{f"{side}_goals": getattr(self, f"{side}_goals") + 1})
        if event == "goal_disallowed":
            return replace(self, **{f"{side}_goals": max(0, getattr(self, f"{side}_goals") - 1)})
        if event == "red_card":
            return replace(self, **{f"{side}_reds": getattr(self, f"{side}_reds") + 1})
        return self


def remaining_minutes(minute: float) -> float:
    """Expected playing time left in regulation, including expected stoppage."""
    if minute < 45:
        return (45 - minute) + STOPPAGE_MINUTES[1] + 45 + STOPPAGE_MINUTES[2]
    return max(90 + STOPPAGE_MINUTES[2] - minute, 0.0)


def outcome_probs(mu_home: float, mu_away: float, lead: int = 0) -> np.ndarray:
    """P(home win, draw, away win) given remaining expected goals and current home lead."""
    k = np.arange(MAX_GOALS + 1)
    ph = poisson.pmf(k, mu_home)
    pa = poisson.pmf(k, mu_away)
    joint = np.outer(ph, pa)  # joint[i, j] = P(home scores i more, away j more)
    diff = k[:, None] - k[None, :] + lead
    p = np.array([joint[diff > 0].sum(), joint[diff == 0].sum(), joint[diff < 0].sum()])
    return p / p.sum()


def fit_rates(p_home: float, p_draw: float, p_away: float) -> tuple[float, float]:
    """Full-match expected goals (home, away) that best reproduce the given 1X2 prices."""
    target = np.array([p_home, p_draw, p_away]) / (p_home + p_draw + p_away)

    def loss(log_mu):
        return float(((outcome_probs(*np.exp(log_mu)) - target) ** 2).sum())

    res = minimize(loss, x0=np.log([1.4, 1.1]), method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-12})
    return tuple(np.exp(res.x))


@dataclass
class InPlayModel:
    lambda_home: float
    lambda_away: float

    @classmethod
    def from_market(cls, p_home: float, p_draw: float, p_away: float) -> "InPlayModel":
        return cls(*fit_rates(p_home, p_draw, p_away))

    @classmethod
    def from_advance_price(cls, p_home_advances: float, total_goals: float = TYPICAL_TOTAL_GOALS) -> "InPlayModel":
        """Fit from a two-way "to advance" price, which pins down only one degree of
        freedom; the other comes from a prior on total expected goals."""

        def gap(share):
            m = cls(total_goals * share, total_goals * (1 - share))
            return m.advance_probs(MatchState())[0] - p_home_advances

        share = brentq(gap, 0.01, 0.99)
        return cls(total_goals * share, total_goals * (1 - share))

    def _rates(self, state: MatchState, minutes: float) -> tuple[float, float]:
        frac = minutes / EFFECTIVE_MINUTES
        mu_h = self.lambda_home * frac * RED_CARD_OWN_FACTOR**state.home_reds * RED_CARD_OPP_FACTOR**state.away_reds
        mu_a = self.lambda_away * frac * RED_CARD_OWN_FACTOR**state.away_reds * RED_CARD_OPP_FACTOR**state.home_reds
        return mu_h, mu_a

    def probs(self, state: MatchState) -> np.ndarray:
        """Regulation-time P(home win, draw, away win)."""
        mu_h, mu_a = self._rates(state, remaining_minutes(state.minute))
        return outcome_probs(mu_h, mu_a, state.home_goals - state.away_goals)

    def probs_dict(self, state: MatchState) -> dict[str, float]:
        return dict(zip(("home", "draw", "away"), self.probs(state)))

    def advance_probs(self, state: MatchState, home_penalty_win: float = 0.5) -> np.ndarray:
        """Knockout P(home advances, away advances): regulation, then 30' extra time
        (red cards carry over), then a shootout won by home with ``home_penalty_win``.
        States inside extra time aren't modelled; ``state`` must be in regulation."""
        reg_h, reg_d, reg_a = self.probs(state)
        et_h, et_d, et_a = outcome_probs(*self._rates(state, EXTRA_TIME_MINUTES))
        home = reg_h + reg_d * (et_h + et_d * home_penalty_win)
        return np.array([home, 1.0 - home])
