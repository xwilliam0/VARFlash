"""Download Kalshi match markets to disk and load them back as tidy DataFrames.

Layout::

    data/kalshi/<SERIES>/<EVENT>/event.json          event + market metadata
    data/kalshi/<SERIES>/<EVENT>/candles_1m.parquet  one row per (market, minute)
    data/kalshi/<SERIES>/<EVENT>/trades.parquet      one row per trade
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from ..paths import KALSHI_DATA, KICKOFFS_CSV
from .client import KalshiClient, MarketRef

log = logging.getLogger(__name__)

# Final whistle minus this is a decent kickoff guess when none is supplied:
# 90' + ~15' half-time + ~7' total stoppage.
DEFAULT_KICKOFF_BEFORE_CLOSE = timedelta(minutes=112)
PRE_MATCH_WINDOW = timedelta(hours=4)
DRAW_SUFFIXES = {"TIE", "DRAW", "D"}


def utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _parse_ts(s: str | None) -> pd.Timestamp | None:
    return utc(s) if s else None


def _flatten_candles(ticker: str, candles: list[dict]) -> pd.DataFrame:
    rows = []
    for c in candles:
        row = {
            "ticker": ticker,
            "ts": pd.Timestamp(c["end_period_ts"], unit="s", tz="UTC"),
            "volume": c.get("volume"),
            "open_interest": c.get("open_interest"),
        }
        for group in ("price", "yes_bid", "yes_ask"):
            for k, v in (c.get(group) or {}).items():
                row[f"{group}_{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    num_cols = [c for c in df.columns if c not in ("ticker", "ts")]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    return df


def _trades_frame(trades: list[dict]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=["ts", "ticker", "yes_price", "count", "taker_side"])
    df = pd.DataFrame(trades)
    out = pd.DataFrame(
        {
            "ts": pd.to_datetime(df["created_time"], utc=True, format="ISO8601"),
            "ticker": df["ticker"],
            "yes_price": pd.to_numeric(df["yes_price_dollars"], errors="coerce"),
            "count": pd.to_numeric(df["count_fp"], errors="coerce"),
            "taker_side": df["taker_side"],
        }
    )
    return out.sort_values("ts").reset_index(drop=True)


def outcome_codes(event_ticker: str, market_tickers: list[str]) -> dict[str, str]:
    """Map each market ticker to 'home', 'away' or 'draw'.

    Match event tickers end in <YY><MON><DD><HOME><AWAY> (e.g. ...26JUL11ARGSUI) and
    each market ticker ends in its outcome code (ARG, SUI, TIE).
    """
    matchup = re.sub(r"^\d{2}[A-Z]{3}\d{2}", "", event_ticker.split("-")[-1])
    roles: dict[str, str] = {}
    for t in market_tickers:
        code = t.split("-")[-1]
        if code in DRAW_SUFFIXES:
            roles[t] = "draw"
        elif matchup.startswith(code):
            roles[t] = "home"
        elif matchup.endswith(code):
            roles[t] = "away"
        else:
            roles[t] = "other"
    return roles


@dataclass
class MatchData:
    event: dict
    markets: list[dict]
    candles: pd.DataFrame
    trades: pd.DataFrame
    roles: dict[str, str] = field(default_factory=dict)
    kickoff: pd.Timestamp | None = None
    kickoff_is_estimate: bool = True
    second_half: pd.Timestamp | None = None

    @property
    def event_ticker(self) -> str:
        return self.event["event_ticker"]

    @property
    def title(self) -> str:
        return self.event.get("title", self.event_ticker)

    @property
    def close_time(self) -> pd.Timestamp:
        return max(_parse_ts(m["close_time"]) for m in self.markets)

    @property
    def result(self) -> str | None:
        """'home' / 'away' / 'draw' once settled, else None."""
        for m in self.markets:
            if m.get("result") == "yes":
                return self.roles.get(m["ticker"])
        return None

    def ticker_for(self, role: str) -> str:
        for t, r in self.roles.items():
            if r == role:
                return t
        raise KeyError(role)

    def settlement(self, ticker: str) -> float | None:
        for m in self.markets:
            if m["ticker"] == ticker and m.get("result") in ("yes", "no"):
                return 1.0 if m["result"] == "yes" else 0.0
        return None


def event_dir(series_ticker: str, event_ticker: str, root: Path = KALSHI_DATA) -> Path:
    return root / series_ticker / event_ticker


def pull_event(
    client: KalshiClient,
    event_ticker: str,
    *,
    with_trades: bool = True,
    refresh: bool = False,
    root: Path = KALSHI_DATA,
) -> Path | None:
    """Download one event's markets, 1-minute candles and trades. Returns its directory."""
    event = client.get_event(event_ticker)
    series = event["series_ticker"]
    out = event_dir(series, event_ticker, root)
    if (out / "event.json").exists() and not refresh:
        log.info("cached %s", event_ticker)
        return out

    markets, historical = client.event_markets(event_ticker)
    if not markets:
        log.warning("no markets for %s", event_ticker)
        return None

    open_ts = min(_parse_ts(m["open_time"]) for m in markets)
    close_ts = max(_parse_ts(m["close_time"]) for m in markets)
    now = pd.Timestamp(datetime.now(timezone.utc))
    end = min(close_ts + timedelta(minutes=10), now)
    start = max(open_ts, close_ts - DEFAULT_KICKOFF_BEFORE_CLOSE - PRE_MATCH_WINDOW)
    if start >= end:
        log.warning("%s has not traded yet", event_ticker)
        return None

    candle_frames, trade_frames = [], []
    for m in markets:
        ref = MarketRef(m["ticker"], series, historical)
        candle_frames.append(_flatten_candles(m["ticker"], client.candlesticks(ref, start, end, 1)))
        if with_trades:
            trade_frames.append(_trades_frame(client.trades(ref, start, end)))

    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "event": event,
        "markets": markets,
        "historical": historical,
        "window": [start.isoformat(), end.isoformat()],
    }
    (out / "event.json").write_text(json.dumps(meta, indent=2))
    pd.concat(candle_frames, ignore_index=True).to_parquet(out / "candles_1m.parquet", index=False)
    if with_trades:
        pd.concat(trade_frames, ignore_index=True).sort_values("ts").to_parquet(
            out / "trades.parquet", index=False
        )
    log.info("saved %s (%d markets, historical=%s)", event_ticker, len(markets), historical)
    return out


def pull_series(
    client: KalshiClient,
    series_ticker: str,
    *,
    settled_only: bool = True,
    with_trades: bool = True,
    refresh: bool = False,
    limit: int | None = None,
    root: Path = KALSHI_DATA,
) -> list[Path]:
    events = client.list_events(series_ticker, status="settled" if settled_only else None)
    if limit:
        events = events[:limit]
    saved = []
    for i, ev in enumerate(events, 1):
        log.info("[%d/%d] %s", i, len(events), ev["event_ticker"])
        try:
            path = pull_event(
                client, ev["event_ticker"], with_trades=with_trades, refresh=refresh, root=root
            )
        except Exception as exc:  # keep going across a long bulk pull
            log.error("failed %s: %s", ev["event_ticker"], exc)
            continue
        if path:
            saved.append(path)
    return saved


def load_match(path: Path | str, kickoff: str | pd.Timestamp | None = None) -> MatchData:
    path = Path(path)
    meta = json.loads((path / "event.json").read_text())
    candles = pd.read_parquet(path / "candles_1m.parquet")
    trades_path = path / "trades.parquet"
    trades = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()
    markets = meta["markets"]
    match = MatchData(
        event=meta["event"],
        markets=markets,
        candles=candles,
        trades=trades,
        roles=outcome_codes(meta["event"]["event_ticker"], [m["ticker"] for m in markets]),
    )
    known_kickoff, known_second_half = known_kickoffs().get(match.event_ticker, (None, None))
    kickoff = kickoff if kickoff is not None else known_kickoff
    match.kickoff = (
        utc(kickoff) if kickoff is not None else match.close_time - DEFAULT_KICKOFF_BEFORE_CLOSE
    )
    match.kickoff_is_estimate = kickoff is None
    match.second_half = utc(known_second_half) if known_second_half is not None else None
    return match


def known_kickoffs(path: Path = KICKOFFS_CSV) -> dict[str, tuple[str | None, str | None]]:
    """Manually verified kickoff times (config/kickoffs.csv) keyed by event ticker."""
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    return {
        r.event_ticker: (r.kickoff_utc or None, r.second_half_utc or None) for r in df.itertuples()
    }


def find_event_dir(event_ticker: str, root: Path = KALSHI_DATA) -> Path:
    hits = list(root.glob(f"*/{event_ticker}"))
    if not hits:
        raise FileNotFoundError(f"{event_ticker} not downloaded; run `varflash kalshi pull-event`")
    return hits[0]
