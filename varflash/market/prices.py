"""Turn raw Kalshi candles/trades into implied match-outcome probabilities."""

from __future__ import annotations

import pandas as pd

from ..kalshi.store import MatchData

ROLES = ("home", "draw", "away")


def quotes(match: MatchData) -> pd.DataFrame:
    """Per-minute bid/ask/mid for each outcome, columns MultiIndex (field, role).

    Candle timestamps are the *end* of each minute, so a row stamped t is only
    knowable at t. Consumers must join backward to avoid look-ahead.
    """
    c = match.candles.copy()
    c["role"] = c["ticker"].map(match.roles)
    c = c[c["role"].isin(ROLES)]
    c["bid"] = c["yes_bid_close"]
    c["ask"] = c["yes_ask_close"]
    c["mid"] = (c["bid"] + c["ask"]) / 2
    c["last"] = c["price_close"]
    wide = c.pivot_table(index="ts", columns="role", values=["bid", "ask", "mid", "last", "volume"])
    full_index = pd.date_range(wide.index.min(), wide.index.max(), freq="1min")
    wide = wide.reindex(full_index)
    # Quotes persist between candles; volume does not.
    for field in ("bid", "ask", "mid", "last"):
        wide[field] = wide[field].ffill()
    wide["volume"] = wide["volume"].fillna(0)
    wide.index.name = "ts"
    return wide


def implied_probs(match: MatchData, source: str = "candles", freq: str = "5s") -> pd.DataFrame:
    """Normalised (overround-free) probabilities for home/draw/away.

    ``source="candles"`` uses minute bid/ask mids; ``source="trades"`` uses the
    last traded price resampled to ``freq`` (finer, but noisier).
    """
    if source == "candles":
        raw = quotes(match)["mid"]
    elif source == "trades":
        t = match.trades.copy()
        t["role"] = t["ticker"].map(match.roles)
        t = t[t["role"].isin(ROLES)]
        raw = (
            t.pivot_table(index="ts", columns="role", values="yes_price", aggfunc="last")
            .resample(freq)
            .last()
            .ffill()
        )
    else:
        raise ValueError(f"unknown source {source!r}")
    raw = raw.reindex(columns=list(ROLES)).dropna()
    return raw.div(raw.sum(axis=1), axis=0)


def pre_match_probs(match: MatchData, minutes_before: int = 5) -> pd.Series:
    """Market consensus just before kickoff, the anchor for the in-play model."""
    probs = implied_probs(match, "candles")
    cutoff = match.kickoff - pd.Timedelta(minutes=minutes_before)
    window = probs[(probs.index <= cutoff) & (probs.index > cutoff - pd.Timedelta(minutes=30))]
    if window.empty:
        window = probs[probs.index <= match.kickoff]
    if window.empty:
        raise ValueError(f"no pre-match quotes for {match.event_ticker}")
    p = window.median()
    return p / p.sum()
