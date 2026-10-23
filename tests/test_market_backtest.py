import numpy as np
import pandas as pd
import pytest

from varflash.backtest import MatchClock, Order, QuoteBook, StrategyConfig, generate_orders, run_orders
from varflash.kalshi.store import MatchData, outcome_codes
from varflash.market import detect_shocks, implied_probs
from varflash.spotting.metrics import average_precision

EVENT = "KXTEST-26JUL11AAABBB"
KICKOFF = pd.Timestamp("2026-07-12T01:00:00Z")


def _synthetic_match(goal_minute=30) -> MatchData:
    """Home goal at ``goal_minute``: home jumps 0.50 -> 0.75 one minute later."""
    ts = pd.date_range(KICKOFF - pd.Timedelta("30min"), KICKOFF + pd.Timedelta("100min"), freq="1min")
    jump = KICKOFF + pd.Timedelta(minutes=goal_minute + 1)
    rows = []
    for t in ts:
        after = t >= jump
        for code, p in (("AAA", 0.75 if after else 0.50), ("TIE", 0.15 if after else 0.28), ("BBB", 0.10 if after else 0.22)):
            rows.append({"ticker": f"{EVENT}-{code}", "ts": t, "yes_bid_close": p - 0.01, "yes_ask_close": p + 0.01, "price_close": p, "volume": 10.0})
    markets = [
        {"ticker": f"{EVENT}-{c}", "close_time": "2026-07-12T02:55:00Z", "result": "yes" if c == "AAA" else "no"}
        for c in ("AAA", "TIE", "BBB")
    ]
    m = MatchData(
        event={"event_ticker": EVENT, "title": "AAA vs BBB"},
        markets=markets,
        candles=pd.DataFrame(rows),
        trades=pd.DataFrame(),
        roles=outcome_codes(EVENT, [mk["ticker"] for mk in markets]),
        kickoff=KICKOFF,
    )
    return m


def test_outcome_codes():
    roles = outcome_codes("KXWCGAME-26JUL11ARGSUI", ["X-ARG", "X-SUI", "X-TIE"])
    assert roles == {"X-ARG": "home", "X-SUI": "away", "X-TIE": "draw"}


def test_shock_detection_finds_the_goal():
    m = _synthetic_match()
    shocks = detect_shocks(implied_probs(m, "candles"), threshold=0.1)
    assert len(shocks) == 1
    s = shocks.iloc[0]
    assert s.role == "home" and s.delta > 0.2
    assert s.peak == KICKOFF + pd.Timedelta(minutes=31)


def test_quotes_never_peek_ahead():
    m = _synthetic_match()
    book = QuoteBook(m)
    jump = KICKOFF + pd.Timedelta(minutes=31)
    assert book.at(f"{EVENT}-AAA", jump - pd.Timedelta(seconds=1)) == pytest.approx((0.49, 0.51))
    assert book.at(f"{EVENT}-AAA", jump) == pytest.approx((0.74, 0.76))


def test_orders_settle_and_pay_fees():
    m = _synthetic_match()
    fills = run_orders(m, [Order(KICKOFF + pd.Timedelta("10min"), f"{EVENT}-AAA", "yes", 100, None)])
    f = fills.iloc[0]
    assert f.entry_px == pytest.approx(0.51) and f.exit_px == 1.0
    assert f.pnl == pytest.approx(100 * 0.49 - f.fees)


def test_fast_goal_signal_is_profitable_slow_one_is_not():
    m = _synthetic_match(goal_minute=30)
    clock = MatchClock(KICKOFF)
    events = pd.DataFrame({"utc": [KICKOFF + pd.Timedelta(minutes=30)], "event": ["goal"], "team": ["home"], "confidence": [1.0]})

    fast = StrategyConfig(processing_latency_s=5, hold=pd.Timedelta("5min"))
    orders, log = generate_orders(m, events, clock, fast)
    assert len(orders) == 1 and orders[0].side == "yes" and log.iloc[0].role == "home"
    assert run_orders(m, orders)["pnl"].sum() > 0

    slow = StrategyConfig(broadcast_delay_s=120, hold=pd.Timedelta("5min"))
    orders, _ = generate_orders(m, events, clock, slow)
    assert not orders or run_orders(m, orders)["pnl"].sum() <= 0


def test_average_precision():
    gts = pd.DataFrame({"video": ["a", "a", "b"], "time_s": [10.0, 100.0, 50.0]})
    perfect = pd.DataFrame({"video": ["a", "a", "b"], "time_s": [11.0, 99.0, 50.0], "confidence": [0.9, 0.8, 0.7]})
    assert average_precision(perfect, gts, 5) == pytest.approx(1.0)
    noisy = pd.concat([perfect, pd.DataFrame({"video": ["a"], "time_s": [500.0], "confidence": [0.95]})])
    assert average_precision(noisy, gts, 5) < 1.0
    assert np.isnan(average_precision(perfect, gts.iloc[0:0], 5))
