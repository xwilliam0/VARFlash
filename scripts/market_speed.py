"""How fast Kalshi match markets reprice on goals / red cards, from the trade tape.

For every abrupt, lasting move (>= ``--min-move`` within 30 s, flat before and after), on the
outcome market that moved most:
  * old / new level: median trade price 40-150 s before and 20-120 s after the move,
  * t_first: first trade >= 20% of the way to the new level ("the market has noticed"),
  * gap_s: time from the last trade at the old level to t_first (quotes pulled, nobody trades),
  * first_frac: how much of the move the first off-level trade already covers,
  * t50 / t90: time from t_first until trades reach 50% / 90% of the move.
Kalshi has no goal timestamps, so this is the market's own speed once it starts moving; the
lag from the ball crossing the line to t_first needs an independent clock (see the summary).

    python scripts/market_speed.py data/kalshi/KXWCGAME/*/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from varflash.paths import OUTPUTS

S = pd.Timedelta(seconds=1)


def jumps(trades: pd.DataFrame, min_move: float) -> list[pd.Timestamp]:
    """Times where the 1 s last-trade price moved >= min_move over 30 s (one per 3-minute cluster)."""
    p = trades.set_index("ts")["yes_price"].resample("1s").last().ffill()
    d = (p - p.shift(30)).abs()
    hits = d[d >= min_move]
    out, last = [], None
    for t in hits.index:
        if last is None or t - last > 180 * S:
            out.append(d.loc[t : t + 60 * S].idxmax())
        last = t
    return out


def measure(trades: pd.DataFrame, t_star: pd.Timestamp, min_move: float) -> dict | None:
    ts, px = trades["ts"], trades["yes_price"]
    pre = px[(ts >= t_star - 150 * S) & (ts <= t_star - 40 * S)]
    post = px[(ts >= t_star + 20 * S) & (ts <= t_star + 120 * S)]
    if len(pre) < 5 or len(post) < 5:
        return None
    p0, p1 = pre.median(), post.median()
    move = p1 - p0
    if abs(move) < min_move or pre.quantile(0.75) - pre.quantile(0.25) > 0.04 \
            or post.quantile(0.75) - post.quantile(0.25) > 0.04 or not 0.02 < p0 < 0.98:
        return None
    w = trades[(ts >= t_star - 60 * S) & (ts <= t_star + 60 * S)]
    x = (w["yes_price"] - p0) / move
    off = np.flatnonzero(x.to_numpy() >= 0.2)
    if not len(off):
        return None
    i = off[0]
    t_first = w["ts"].iloc[i]
    before = w.iloc[:i]
    t_old = before["ts"].iloc[-1] if len(before) else pd.NaT

    def reach(f):
        k = np.flatnonzero(x.to_numpy()[i:] >= f)
        return (w["ts"].iloc[i + k[0]] - t_first).total_seconds() if len(k) else np.nan

    first5 = w[(w["ts"] >= t_first) & (w["ts"] < t_first + 5 * S)]
    pre_rate = len(trades[(ts >= t_first - 60 * S) & (ts < t_first)]) / 60
    return {"t_first": t_first, "p_old": p0, "p_new": p1, "move": move, "first_frac": float(x.iloc[i]),
            "gap_s": (t_first - t_old).total_seconds() if pd.notna(t_old) else np.nan,
            "t50_s": reach(0.5), "t90_s": reach(0.9),
            "trades_first5s": len(first5), "contracts_first5s": float(first5["count"].sum()),
            "trades_per_s_before": pre_rate, "path": (w["ts"] - t_first).dt.total_seconds().to_numpy(),
            "progress": x.to_numpy()}


def analyse_event(d: Path, min_move: float) -> list[dict]:
    tr = pd.read_parquet(d / "trades.parquet")
    found = []
    for ticker, t in tr.groupby("ticker"):
        t = t.sort_values("ts").reset_index(drop=True)
        for t_star in jumps(t, min_move):
            m = measure(t, t_star, min_move)
            if m:
                found.append({"event": d.name, "ticker": ticker, **m})
    # One event moves all outcomes; keep the outcome that moved most, earliest t_first across them.
    found.sort(key=lambda r: r["t_first"])
    out = []
    for r in found:
        if out and r["t_first"] - out[-1]["t_first"] <= 60 * S:
            first = min(out[-1]["first_any"], r["t_first"])
            if abs(r["move"]) > abs(out[-1]["move"]):
                out[-1] = {**r, "first_any": first}
            else:
                out[-1]["first_any"] = first
        else:
            out.append({**r, "first_any": r["t_first"]})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("events", nargs="+")
    p.add_argument("--min-move", type=float, default=0.10)
    p.add_argument("--out", default=str(OUTPUTS / "market_speed"))
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in map(Path, a.events):
        if (d / "trades.parquet").exists():
            rows += analyse_event(d, a.min_move)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no repricings found")
        return
    df["lead_other_outcomes_s"] = (df["t_first"] - df["first_any"]).dt.total_seconds()
    cols = ["event", "ticker", "t_first", "p_old", "p_new", "move", "gap_s", "first_frac", "t50_s", "t90_s",
            "trades_first5s", "contracts_first5s", "trades_per_s_before"]
    df[cols].to_csv(out / "repricings.csv", index=False)

    print(f"{len(df)} repricings >= {a.min_move:.0%} in {df['event'].nunique()} matches\n")
    q = df[["gap_s", "first_frac", "t50_s", "t90_s", "trades_first5s", "trades_per_s_before"]]
    print(q.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(2).to_string())
    big = df[df["move"].abs() >= 0.2]
    print(f"\nmoves >= 20%: n={len(big)}  median t50 {big.t50_s.median():.1f}s  t90 {big.t90_s.median():.1f}s  "
          f"first trade covers {big.first_frac.median():.0%} of the move")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for _, r in df.iterrows():
        ax[0].plot(r["path"], r["progress"], color="tab:blue", alpha=0.15, lw=0.8)
    grid = np.arange(-30, 61)
    med = np.nanmedian([np.interp(grid, r["path"], r["progress"]) for _, r in df.iterrows()], axis=0)
    ax[0].plot(grid, med, color="k", lw=2, label="median")
    ax[0].axhline(0.5, ls=":", c="gray"), ax[0].axhline(0.9, ls=":", c="gray")
    ax[0].set(xlim=(-30, 60), ylim=(-0.3, 1.3), xlabel="seconds from first off-level trade",
              ylabel="fraction of the move", title="Kalshi repricing paths (trade prices)")
    ax[0].legend()
    ax[1].hist([df.t50_s.clip(upper=60), df.t90_s.clip(upper=60)], bins=np.arange(0, 61, 2),
               label=["to 50% of move", "to 90% of move"])
    ax[1].set(xlabel="seconds after first off-level trade (clipped at 60)", ylabel="repricings",
              title="Time to reprice")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "repricing_speed.png", dpi=130)
    print(f"-> {out}/repricings.csv, {out}/repricing_speed.png")


if __name__ == "__main__":
    main()
