"""How early does the AI VAR need to be? Sweep the time a red-card signal fires.

Embolo's second yellow (Argentina vs Switzerland, 72') was decided by a VAR
review. For each candidate signal time across the review window, run the
strategy and report the trade and its 5-minute PnL, for both the
regulation-time market and the "to advance" market (extra time + penalties).

    python examples/signal_timing_sweep.py
"""

from __future__ import annotations

import pandas as pd

from varflash.backtest import MatchClock, StrategyConfig, generate_orders, run_orders
from varflash.backtest.engine import QuoteBook
from varflash.kalshi.store import find_event_dir, load_match

MARKETS = ["KXWCADVANCE-26JUL11ARGSUI", "KXWCGAME-26JUL11ARGSUI"]
SECOND_HALF_KICKOFF = "2026-07-12T02:06:00Z"  # approximate; only affects the model's match minute
CONTEXT = [  # score state before the red card
    ("2026-07-12T01:09:30Z", "goal", "home"),
    ("2026-07-12T02:27:50Z", "goal", "away"),
]
# The dive happened just before the market starts drifting; the exact second
# needs the video. Verdict times are swept from there to after the card.
INCIDENT = pd.Timestamp("2026-07-12T02:30:30Z")
SWEEP = pd.date_range(INCIDENT, "2026-07-12T02:33:00Z", freq="15s")


def main():
    for event_ticker in MARKETS:
        m = load_match(find_event_dir(event_ticker))
        book = QuoteBook(m)
        clock = MatchClock(m.kickoff, pd.Timestamp(SECOND_HALF_KICKOFF))
        cfg = StrategyConfig(processing_latency_s=0, hold=pd.Timedelta("5min"))
        kind = "to advance" if m.is_advance_market else "regulation 1X2"
        print(f"\n{event_ticker} ({kind})")
        print("signal    minute  trade        price  fair   edge   contracts  pnl_5min")
        for t in SWEEP:
            events = pd.DataFrame(
                [(pd.Timestamp(u), pd.NaT, e, team, 1.0) for u, e, team in CONTEXT]
                + [(t, INCIDENT, "red_card", "away", 0.9)],
                columns=["utc", "incident_utc", "event", "team", "confidence"],
            )
            orders, log = generate_orders(m, events, clock, cfg, book=book)
            d = log.iloc[-1]
            red = [o for o in orders if o.reason.startswith("red_card")]
            pnl = run_orders(m, red, book)["pnl"].sum() if red else 0.0
            team = m.ticker_for(d.role).split("-")[-1]
            print(
                f"{t:%H:%M:%S}  {d.minute:5.1f}  {team:>4} {d.side:<4}  {d.price:6.2f}  "
                f"{d['fair_' + d.role]:.2f}  {d.edge:+.3f}  {d.contracts:9d}  {pnl:+8.2f}"
            )


if __name__ == "__main__":
    main()
