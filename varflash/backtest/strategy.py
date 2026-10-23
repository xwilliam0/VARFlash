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

from ..kalshi.store import MatchData, find_regulation_sibling
from ..market.prices import pre_match_probs
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
    # Only trade in the direction the event pushes each outcome. Otherwise, once
    # the market has reacted, size disagreements with the model turn into fades.
    with_event_only: bool = True


def video_events_to_utc(events: pd.DataFrame, clock: MatchClock) -> pd.DataFrame:
    """Accepts columns ``video_s`` or ``game_time`` ("2 - 13:20") or ``utc``.

    ``utc`` is when the model's verdict is ready. For reviewed decisions (VAR),
    optionally give ``incident_utc`` / ``incident_video_s``: when the incident
    itself happened, before anyone could react. The market price at that moment
    is the anchor, so a verdict that arrives after the market has already moved
    shows no edge instead of double-counting the event.
    """
    ev = events.copy()
    if "incident_video_s" in ev:
        ev["incident_utc"] = ev["incident_video_s"].map(
            lambda s: clock.video_to_utc(s) if pd.notna(s) else pd.NaT
        )
    if "incident_utc" in ev:
        ev["incident_utc"] = pd.to_datetime(ev["incident_utc"], utc=True)
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
    """Normalised mids for ``match.outcome_roles`` as of ``ts``."""
    mids = np.array([sum(book.at(match.ticker_for(r), ts)) / 2 for r in match.outcome_roles])
    return mids / mids.sum()


def build_model(match: MatchData) -> InPlayModel:
    """Fit scoring rates from pre-match prices.

    "To advance" markets only give one number, so prefer the same game's
    regulation 1X2 market when it has been downloaded.
    """
    if not match.is_advance_market:
        pm = pre_match_probs(match)
        return InPlayModel.from_market(pm["home"], pm["draw"], pm["away"])
    sibling = find_regulation_sibling(match)
    if sibling is not None:
        return build_model(sibling)
    return InPlayModel.from_advance_price(pre_match_probs(match)["home"])


def model_probs(match: MatchData, model: InPlayModel, state: MatchState) -> np.ndarray:
    return model.advance_probs(state) if match.is_advance_market else model.probs(state)


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
    model = model or build_model(match)
    roles = match.outcome_roles

    state = MatchState()
    orders: list[Order] = []
    log = []
    for ev in events.itertuples():
        if ev.event not in EVENT_TYPES:
            continue
        minute = clock.match_minute(ev.utc)
        before = MatchState(minute, state.home_goals, state.away_goals, state.home_reds, state.away_reds)
        after = before.apply(ev.event, ev.team)
        p_before, p_after = model_probs(match, model, before), model_probs(match, model, after)
        jump = ev.confidence * (p_after - p_before)
        if cfg.anchor == "market":
            incident = getattr(ev, "incident_utc", pd.NaT)
            anchor_ts = ev.utc if pd.isna(incident) else incident
            fair = market_probs(match, book, anchor_ts) + jump
            fair = np.clip(fair, 0.01, 0.99)
            fair = fair / fair.sum()
        else:
            fair = p_before + jump
        if ev.confidence >= cfg.min_confidence:
            state = after

        signal_ts = ev.utc + pd.Timedelta(seconds=cfg.broadcast_delay_s + cfg.processing_latency_s)
        best = None
        for role, p_yes, move in zip(roles, fair, jump):
            ticker = match.ticker_for(role)
            for side, p_win in (("yes", p_yes), ("no", 1 - p_yes)):
                if cfg.with_event_only and (move > 0) != (side == "yes"):
                    continue
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
                **{f"fair_{r}": round(p, 3) for r, p in zip(roles, fair)},
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
