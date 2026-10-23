"""Read-only client for Kalshi's public market-data API.

Only unauthenticated GET endpoints are used; this project never places orders.
Settled markets older than the exchange's "historical cutoff" are served from
the /historical/* endpoints, so every market fetch records which API it came
from and later candle/trade requests are routed accordingly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

import requests

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Kalshi caps a single candlestick response, so long ranges are chunked.
MAX_CANDLES_PER_REQUEST = 4000


class KalshiError(RuntimeError):
    pass


@dataclass(frozen=True)
class MarketRef:
    ticker: str
    series_ticker: str
    historical: bool


def to_unix(ts: datetime | str | int | float) -> int:
    if isinstance(ts, (int, float)):
        return int(ts)
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return int(ts.timestamp())


class KalshiClient:
    def __init__(
        self,
        base_url: str = BASE_URL,
        max_requests_per_sec: float = 8.0,
        max_retries: int = 5,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.min_interval = 1.0 / max_requests_per_sec
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"
        self._last_request = 0.0

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        params = {k: v for k, v in (params or {}).items() if v is not None}
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise KalshiError(f"GET {url} failed: {exc}") from exc
                time.sleep(2**attempt)
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == self.max_retries:
                    break
                retry_after = resp.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after else 2**attempt)
                continue
            if resp.status_code == 403:
                raise KalshiError(
                    f"403 Forbidden for {url}. Kalshi may be geofencing your IP; "
                    "see README for running the pull from a cloud VM."
                )
            if not resp.ok:
                raise KalshiError(f"GET {url} -> {resp.status_code}: {resp.text[:300]}")
            return resp.json()
        raise KalshiError(f"GET {url} exhausted retries (last status {resp.status_code})")

    def paginate(self, path: str, key: str, params: dict[str, Any] | None = None) -> Iterator[dict]:
        params = dict(params or {})
        while True:
            page = self.get(path, params)
            yield from page.get(key) or []
            cursor = page.get("cursor")
            if not cursor:
                return
            params["cursor"] = cursor

    # ---- catalog -----------------------------------------------------------

    def historical_cutoff(self) -> dict[str, str]:
        return self.get("historical/cutoff")

    def list_series(self, category: str | None = None) -> list[dict]:
        return self.get("series", {"category": category}).get("series") or []

    def list_events(self, series_ticker: str, status: str | None = None) -> list[dict]:
        return list(
            self.paginate(
                "events", "events", {"series_ticker": series_ticker, "status": status, "limit": 200}
            )
        )

    def get_event(self, event_ticker: str) -> dict:
        return self.get(f"events/{event_ticker}")["event"]

    def event_markets(self, event_ticker: str) -> tuple[list[dict], bool]:
        """Markets for an event, plus whether they came from the historical archive."""
        live = list(self.paginate("markets", "markets", {"event_ticker": event_ticker, "limit": 200}))
        if live:
            return live, False
        hist = list(
            self.paginate("historical/markets", "markets", {"event_ticker": event_ticker, "limit": 200})
        )
        return hist, True

    # ---- price data --------------------------------------------------------

    def candlesticks(
        self,
        market: MarketRef,
        start: datetime | str | int,
        end: datetime | str | int,
        period_minutes: int = 1,
    ) -> list[dict]:
        if period_minutes not in (1, 60, 1440):
            raise ValueError("Kalshi supports 1, 60 or 1440 minute candles")
        start_ts, end_ts = to_unix(start), to_unix(end)
        if market.historical:
            path = f"historical/markets/{market.ticker}/candlesticks"
        else:
            path = f"series/{market.series_ticker}/markets/{market.ticker}/candlesticks"
        step = MAX_CANDLES_PER_REQUEST * period_minutes * 60
        out: list[dict] = []
        for chunk_start in range(start_ts, end_ts, step):
            chunk_end = min(chunk_start + step, end_ts)
            page = self.get(
                path,
                {"start_ts": chunk_start, "end_ts": chunk_end, "period_interval": period_minutes},
            )
            out.extend(page.get("candlesticks") or [])
        return out

    def trades(
        self,
        market: MarketRef,
        start: datetime | str | int | None = None,
        end: datetime | str | int | None = None,
    ) -> list[dict]:
        path = "historical/trades" if market.historical else "markets/trades"
        params = {
            "ticker": market.ticker,
            "limit": 1000,
            "min_ts": to_unix(start) if start is not None else None,
            "max_ts": to_unix(end) if end is not None else None,
        }
        return list(self.paginate(path, "trades", params))
