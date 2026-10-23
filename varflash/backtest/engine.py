"""Event-level backtester for Kalshi binary contracts.

Fills are simulated as marketable (taker) orders against the most recent
quote *at or before* the decision time, so nothing from the future leaks in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..kalshi.store import MatchData
from .fees import kalshi_fee

MIN_PRICE, MAX_PRICE = 0.01, 0.99


class QuoteBook:
    """Best bid/ask per ticker over time, refined by the last trade when it's fresher."""

    def __init__(self, match: MatchData, use_trades: bool = True):
        self.match = match
        self._candles = {
            t: g.set_index("ts")[["yes_bid_close", "yes_ask_close"]].sort_index().dropna()
            for t, g in match.candles.groupby("ticker")
        }
        self._trades = {}
        if use_trades and not match.trades.empty:
            self._trades = {
                t: g.set_index("ts")["yes_price"].sort_index()
                for t, g in match.trades.groupby("ticker")
            }

    def at(self, ticker: str, ts: pd.Timestamp) -> tuple[float, float]:
        candles = self._candles[ticker]
        i = candles.index.searchsorted(ts, side="right") - 1
        if i < 0:
            raise LookupError(f"no quote for {ticker} before {ts}")
        candle_ts = candles.index[i]
        bid, ask = candles.iloc[i]
        trades = self._trades.get(ticker)
        if trades is not None:
            j = trades.index.searchsorted(ts, side="right") - 1
            if j >= 0 and trades.index[j] > candle_ts:
                half_spread = max((ask - bid) / 2, 0.005)
                last = trades.iloc[j]
                bid = float(np.clip(last - half_spread, MIN_PRICE, MAX_PRICE))
                ask = float(np.clip(last + half_spread, MIN_PRICE, MAX_PRICE))
        return float(bid), float(ask)

    def entry_price(self, ticker: str, side: str, ts: pd.Timestamp) -> float:
        bid, ask = self.at(ticker, ts)
        return ask if side == "yes" else 1.0 - bid

    def exit_price(self, ticker: str, side: str, ts: pd.Timestamp) -> float:
        bid, ask = self.at(ticker, ts)
        return bid if side == "yes" else 1.0 - ask


@dataclass
class Order:
    ts: pd.Timestamp
    ticker: str
    side: str  # "yes" or "no"
    contracts: int
    hold: pd.Timedelta | None = None  # None = hold to settlement
    reason: str = ""


def run_orders(match: MatchData, orders: list[Order], book: QuoteBook | None = None) -> pd.DataFrame:
    book = book or QuoteBook(match)
    rows = []
    for o in orders:
        entry = book.entry_price(o.ticker, o.side, o.ts)
        entry_fee = kalshi_fee(o.contracts, entry)
        if o.hold is None:
            settle_yes = match.settlement(o.ticker)
            if settle_yes is None:
                raise ValueError(f"{o.ticker} not settled; give the order a hold period")
            exit_ts, exit_fee = match.close_time, 0.0
            exit_px = settle_yes if o.side == "yes" else 1.0 - settle_yes
        else:
            exit_ts = o.ts + o.hold
            exit_px = book.exit_price(o.ticker, o.side, exit_ts)
            exit_fee = kalshi_fee(o.contracts, exit_px)
        pnl = o.contracts * (exit_px - entry) - entry_fee - exit_fee
        rows.append(
            {
                "entry_ts": o.ts,
                "exit_ts": exit_ts,
                "ticker": o.ticker,
                "side": o.side,
                "contracts": o.contracts,
                "entry_px": entry,
                "exit_px": exit_px,
                "fees": entry_fee + exit_fee,
                "pnl": pnl,
                "return": pnl / (o.contracts * entry) if o.contracts else 0.0,
                "reason": o.reason,
            }
        )
    return pd.DataFrame(rows)


def summarize(fills: pd.DataFrame) -> dict:
    if fills.empty:
        return {"trades": 0, "pnl": 0.0}
    capital = float((fills["contracts"] * fills["entry_px"]).sum())
    pnl = float(fills["pnl"].sum())
    return {
        "trades": len(fills),
        "pnl": round(pnl, 2),
        "capital_deployed": round(capital, 2),
        "return_on_capital": round(pnl / capital, 4) if capital else 0.0,
        "hit_rate": round(float((fills["pnl"] > 0).mean()), 3),
        "fees": round(float(fills["fees"].sum()), 2),
    }


def markouts(
    signals: pd.DataFrame,
    probs: pd.DataFrame,
    horizons=("30s", "1min", "2min", "5min"),
) -> pd.DataFrame:
    """How the market price of the signalled outcome moved after each signal.

    Positive markout = the market moved in the signal's direction afterwards,
    i.e. the signal was ahead of the market. This is the core PoC metric.
    """
    probs = probs.sort_index()
    out = signals.copy()
    sign = np.where(out["side"] == "yes", 1.0, -1.0)
    base = np.array([probs[r].asof(t) for r, t in zip(out["role"], out["ts"])])
    out["p_at_signal"] = base
    for h in horizons:
        later = np.array(
            [probs[r].asof(t + pd.Timedelta(h)) for r, t in zip(out["role"], out["ts"])]
        )
        out[f"markout_{h}"] = sign * (later - base)
    return out
