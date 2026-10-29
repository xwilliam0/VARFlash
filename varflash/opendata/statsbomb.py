"""StatsBomb open event data: cards, fouls and goals for priors and quant sanity checks.

https://github.com/statsbomb/open-data (free for research, attribution required).
Used here to (1) measure how often fouls become cards (a prior for the card
detector) and (2) re-estimate the red-card scoring factors used by ``quant/poisson.py``.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..paths import DATA

log = logging.getLogger(__name__)

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
ROOT = DATA / "statsbomb"
# (competition_id, season_id): recent men's tournaments + one full league season.
DEFAULT_SEASONS = {
    "FIFA World Cup 2022": (43, 106), "FIFA World Cup 2018": (43, 3), "UEFA Euro 2024": (55, 282),
    "UEFA Euro 2020": (55, 43), "Copa America 2024": (223, 282), "African Cup of Nations 2023": (1267, 107),
    "Premier League 2015/2016": (2, 27),
}
CARDS = {"Yellow Card": "yellow", "Second Yellow": "second_yellow", "Red Card": "red"}


def _get(rel: str, root: Path = ROOT):
    path = root / rel
    if not path.exists():
        r = requests.get(f"{BASE}/{rel}", timeout=60)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    return json.loads(path.read_text())


def matches(seasons: dict[str, tuple[int, int]] = DEFAULT_SEASONS) -> pd.DataFrame:
    rows = []
    for name, (cid, sid) in seasons.items():
        for m in _get(f"matches/{cid}/{sid}.json"):
            rows.append({"match_id": m["match_id"], "competition": name, "date": m["match_date"],
                         "home": m["home_team"]["home_team_name"], "away": m["away_team"]["away_team_name"],
                         "home_score": m["home_score"], "away_score": m["away_score"]})
    return pd.DataFrame(rows)


def _minute(e: dict) -> float:
    return e["minute"] + e["second"] / 60


def match_events(match_id: int) -> pd.DataFrame:
    """Fouls, cards and goals of one match (regulation + extra time, no shoot-outs)."""
    rows = []
    for e in _get(f"events/{match_id}.json"):
        if e["period"] > 4:
            continue
        t, team = e["type"]["name"], e.get("team", {}).get("name")
        base = {"match_id": match_id, "period": e["period"], "minute": _minute(e), "team": team}
        card = (e.get("foul_committed") or e.get("bad_behaviour") or {}).get("card", {}).get("name")
        if t == "Foul Committed":
            rows.append({**base, "kind": "foul", "card": CARDS.get(card)})
        elif t == "Bad Behaviour" and card:
            rows.append({**base, "kind": "card_no_foul", "card": CARDS.get(card)})
        elif t == "Shot" and e["shot"]["outcome"]["name"] == "Goal":
            rows.append({**base, "kind": "goal", "card": None})
        elif t == "Own Goal For":
            rows.append({**base, "kind": "goal", "card": None})
    return pd.DataFrame(rows, columns=["match_id", "period", "minute", "team", "kind", "card"])


def load_events(seasons=DEFAULT_SEASONS, workers: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    ms = matches(seasons)
    with ThreadPoolExecutor(workers) as pool:
        evs = list(pool.map(match_events, ms["match_id"]))
    log.info("%d matches, %d competitions", len(ms), ms["competition"].nunique())
    return ms, pd.concat(evs, ignore_index=True)


def card_summary(ms: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Per competition: cards per match and how often a foul draws each card."""
    ev = ev.merge(ms[["match_id", "competition"]], on="match_id")
    rows = []
    for comp, g in list(ev.groupby("competition")) + [("ALL", ev)]:
        n = g["match_id"].nunique()
        fouls = g[g["kind"] == "foul"]
        cards = g[g["card"].notna()]
        rows.append({"competition": comp, "matches": n, "fouls_per_match": len(fouls) / n,
                     "yellow_per_match": (cards["card"] == "yellow").sum() / n,
                     "sending_off_per_match": cards["card"].isin(["red", "second_yellow"]).sum() / n,
                     "P(yellow|foul)": (fouls["card"] == "yellow").mean(),
                     "P(sending_off|foul)": fouls["card"].isin(["red", "second_yellow"]).mean(),
                     "share_of_cards_without_foul": (cards["kind"] == "card_no_foul").mean()})
    return pd.DataFrame(rows)


def red_card_effect(ms: pd.DataFrame, ev: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Scoring-rate multipliers after the first sending-off, for the 10-man team and its opponent.

    Observed goals after the red vs. the goals the same team would be expected to score in
    that stretch of the match, using the minute-by-minute scoring profile of all matches
    (late minutes are busier) scaled by the team's own pre-red scoring vs. the average
    pre-red scoring. Pooled over matches; 95% CI by bootstrapping matches.
    """
    reg = ev[ev["period"] <= 2]
    goals = reg[reg["kind"] == "goal"]
    n_matches = reg["match_id"].nunique()
    # Goals per team-minute, by minute of match (0..95), smoothed.
    hist = np.bincount(np.clip(goals["minute"].astype(int), 0, 95), minlength=96) / (2 * n_matches)
    profile = np.convolve(hist, np.ones(9) / 9, mode="same")
    cum = np.concatenate([[0], np.cumsum(profile)])

    def expected(a: float, b: float) -> float:
        return float(np.interp(b, np.arange(97), cum) - np.interp(a, np.arange(97), cum))

    rows = []
    for mid, g in reg.groupby("match_id"):
        reds = g[g["card"].isin(["red", "second_yellow"])].sort_values("minute")
        if reds.empty:
            continue
        t_red, carded = float(reds["minute"].iloc[0]), reds["team"].iloc[0]
        teams = [t for t in g["team"].dropna().unique() if t != carded]
        if t_red > 85 or not teams:
            continue
        opp = teams[0]
        end = max(90.0, float(g["minute"].max()))
        for side, team in (("own", carded), ("opp", opp)):
            tg = goals[(goals["match_id"] == mid) & (goals["team"] == team)]
            rows.append({"match_id": mid, "side": side, "minute": t_red,
                         "goals_after": int((tg["minute"] >= t_red).sum()), "exp_after": expected(t_red, end),
                         "goals_before": int((tg["minute"] < t_red).sum()), "exp_before": expected(0, t_red)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    rng = np.random.default_rng(seed)
    out = []
    for side, g in df.groupby("side"):
        def ratio(x):
            # (after / expected after) relative to (before / expected before), with a prior of
            # one expected-goal's worth of "normal" scoring before the red to tame tiny samples.
            before = (x["goals_before"].sum() + 1) / (x["exp_before"].sum() + 1)
            return x["goals_after"].sum() / x["exp_after"].sum() / before

        ids = g["match_id"].unique()
        boot = [ratio(g.set_index("match_id").loc[rng.choice(ids, len(ids))]) for _ in range(n_boot)]
        out.append({"side": side, "matches": len(ids), "goals_after": int(g["goals_after"].sum()),
                    "expected_after": g["exp_after"].sum(), "factor": ratio(g),
                    "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5)})
    return pd.DataFrame(out)
