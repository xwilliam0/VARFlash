from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ..kalshi.store import MatchData  # noqa: E402

COLORS = {"home": "#1f77b4", "draw": "#7f7f7f", "away": "#d62728"}


def plot_match(
    match: MatchData,
    probs: pd.DataFrame,
    shocks: pd.DataFrame | None = None,
    signals: pd.DataFrame | None = None,
    out: Path | str | None = None,
):
    names = {
        "home": match.ticker_for("home").split("-")[-1],
        "away": match.ticker_for("away").split("-")[-1],
        "draw": "Draw",
    }
    start = match.kickoff - pd.Timedelta(minutes=20)
    view = probs[probs.index >= start]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    for role in ("home", "draw", "away"):
        if role in view:
            ax.plot(view.index, view[role], color=COLORS[role], lw=1.4, label=names[role])
    ax.axvline(match.kickoff, color="k", ls="--", lw=0.8)
    label = " kickoff (est.)" if match.kickoff_is_estimate else " kickoff"
    ax.text(match.kickoff, 1.01, label, fontsize=8, transform=ax.get_xaxis_transform())

    if shocks is not None:
        for s in shocks.itertuples():
            ax.axvspan(s.onset, s.peak, color="orange", alpha=0.25)
            ax.annotate(
                f"{names.get(s.role, s.role)} {s.delta:+.0%}",
                (s.peak, s.p_after),
                textcoords="offset points",
                xytext=(4, 6),
                fontsize=8,
            )
    if signals is not None and not signals.empty:
        for s in signals.itertuples():
            ax.axvline(s.ts, color="green", lw=1, alpha=0.8)

    ax.set_ylim(0, 1)
    ax.set_ylabel("implied probability (normalised)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz="UTC"))
    ax.set_xlabel("UTC")
    result = match.result
    ax.set_title(f"{match.title}  |  result: {names.get(result, result)}")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130)
        plt.close(fig)
    return fig
