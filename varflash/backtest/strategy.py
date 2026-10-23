"""Turn video-detected match events into Kalshi orders.

For each event the vision stack reports (goal, red card, goal overturned by
VAR, ...) with a confidence, the in-play Poisson model gives the fair 1X2
probabilities before and after it. The confidence-weighted fair value is
compared to the quote at the moment the signal would have been actionable;
if the edge after fees clears a threshold, a fractional-Kelly order is sent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..kalshi.store import MatchData
from ..market.prices import ROLES, pre_match_probs
from ..quant.poisson import InPlayModel, MatchState
from .clock import MatchClock, parse_game_time
from .engine import Order, QuoteBook
from .fees import fee_per_contract

EVENT_TYPES = {"goal", "goal_disallowed", "red_card"}


@dataclass
class StrategyConfig:
    broadcast_delay_s: float = 0.0  # how far the video feed lags the stadium
    processing_latency_s: float = 2.0  # model inference + order round trip
    min_edge: float = 0.03  # required edge per contract after fees, in dollars
    kelly_fraction: float = 0.25
    bankroll: float = 1000.0
    max_contracts: int = 500
    hold: pd.Timedelta | None = field(default_factory=lambda: pd.Timedelta(minutes=5))
    min_confidence: float = 0.5  # commit an event to the match state above this
    # "market": fair = market price just before the event + model-implied jump,
    #   so we only bet on reaction speed, not on out-guessing the market's level.
    # "model": fair = raw Poisson model probabilities.
    anchor: str = "market"


def video_events_to_utc(events: pd.DataFrame, clock: MatchClock) -> pd.DataFrame:
    """Accepts columns ``video_s`` or ``game_time`` ("2 - 13:20") or ``utc``."""
    ev = events.copy()
    if "utc" in ev:
        ev["utc"] = pd.to_datetime(ev["utc"], utc=True)
    elif "video_s" in ev:
        ev["utc"] = ev["video_s"].map(clock.video_to_utc)
    elif "game_time" in ev:
        ev["utc"] = ev["game_time"].map(lambda s: clock.game_time_to_utc(*parse_game_time(s)))
    else:
        raise ValueError("events need a utc, video_s or game_time column")
    if "confidence" not in ev:
        ev["confidence"] = 1.0
    return ev.sort_values("utc").reset_index(drop=True)


def market_probs(match: MatchData, book: QuoteBook, ts: pd.Timestamp) -> np.ndarray:
    """Normalised home/draw/away mids as of ``ts``."""
    mids = np.array([sum(book.at(match.ticker_for(r), ts)) / 2 for r in ROLES])
    return mids / mids.sum()


def kelly_contracts(p_win: float, price: float, cfg: StrategyConfig) -> int:
    """Fractional Kelly size for a binary contract bought at ``price`` that wins with ``p_win``."""
    if not 0 < price < 1:
        return 0
    f_star = (p_win - price) / (1 - price)
    if f_star <= 0:
        return 0
    stake = cfg.kelly_fraction * f_star * cfg.bankroll
    return int(min(math.floor(stake / price), cfg.max_contracts))


def generate_orders(
    match: MatchData,
    events: pd.DataFrame,
    clock: MatchClock,
    cfg: StrategyConfig | None = None,
    model: InPlayModel | None = None,
    book: QuoteBook | None = None,
) -> tuple[list[Order], pd.DataFrame]:
    """Returns (orders, decision log). ``events`` must already have a ``utc`` column."""
    cfg = cfg or StrategyConfig()
    book = book or QuoteBook(match)
    if model is None:
        pm = pre_match_probs(match)
        model = InPlayModel.from_market(pm["home"], pm["draw"], pm["away"])

    state = MatchState()
    orders: list[Order] = []
    log = []
    for ev in events.itertuples():
        if ev.event not in EVENT_TYPES:
            continue
        minute = clock.match_minute(ev.utc)
        before = MatchState(minute, state.home_goals, state.away_goals, state.home_reds, state.away_reds)
        after = before.apply(ev.event, ev.team)
        p_before, p_after = model.probs(before), model.probs(after)
        jump = ev.confidence * (p_after - p_before)
        if cfg.anchor == "market":
            fair = market_probs(match, book, ev.utc) + jump
            fair = np.clip(fair, 0.01, 0.99)
            fair = fair / fair.sum()
        else:
            fair = p_before + jump
        if ev.confidence >= cfg.min_confidence:
            state = after

        signal_ts = ev.utc + pd.Timedelta(seconds=cfg.broadcast_delay_s + cfg.processing_latency_s)
        best = None
        for role, p_yes in zip(ROLES, fair):
            ticker = match.ticker_for(role)
            for side, p_win in (("yes", p_yes), ("no", 1 - p_yes)):
                price = book.entry_price(ticker, side, signal_ts)
                edge = p_win - price - fee_per_contract(price)
                if best is None or edge > best[0]:
                    best = (edge, role, ticker, side, price, p_win)

        edge, role, ticker, side, price, p_win = best
        n = kelly_contracts(p_win, price, cfg) if edge >= cfg.min_edge else 0
        log.append(
            {
                "event_utc": ev.utc,
                "ts": signal_ts,
                "minute": round(minute, 1),
                "event": ev.event,
                "team": ev.team,
                "confidence": ev.confidence,
                **{f"fair_{r}": round(p, 3) for r, p in zip(ROLES, fair)},
                "role": role,
                "side": side,
                "price": price,
                "edge": round(edge, 4),
                "contracts": n,
            }
        )
        if n > 0:
            orders.append(
                Order(signal_ts, ticker, side, n, cfg.hold, reason=f"{ev.event}:{ev.team}@{minute:.0f}'")
            )
    return orders, pd.DataFrame(log)
