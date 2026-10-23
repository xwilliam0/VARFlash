"""Find sudden repricings ("shocks") in a match's implied probabilities.

Shocks mark where something decisive happened on the pitch (goal, red card,
VAR overturn). They are the targets a video model has to beat: a signal is
only useful if it arrives before the shock is priced in.
"""

from __future__ import annotations

import pandas as pd


def detect_shocks(
    probs: pd.DataFrame,
    window: str = "2min",
    threshold: float = 0.08,
    min_gap: str = "3min",
) -> pd.DataFrame:
    """Return one row per shock with its onset, peak move and the outcome that moved most.

    ``probs`` is a time-indexed frame of outcome probabilities (e.g. from
    ``implied_probs``). A shock is any stretch where some outcome moved by more
    than ``threshold`` within ``window``; stretches closer than ``min_gap`` merge.
    """
    probs = probs.sort_index()
    lagged = probs.reindex(probs.index - pd.Timedelta(window), method="ffill")
    lagged.index = probs.index
    delta = probs - lagged
    biggest = delta.abs().max(axis=1)
    flagged = biggest[biggest > threshold]
    if flagged.empty:
        return pd.DataFrame(
            columns=["onset", "peak", "role", "p_before", "p_after", "delta", "magnitude"]
        )

    cluster_id = (flagged.index.to_series().diff() > pd.Timedelta(min_gap)).cumsum()
    rows = []
    for _, idx in flagged.groupby(cluster_id.values).groups.items():
        peak = biggest.loc[idx].idxmax()
        role = delta.loc[peak].abs().idxmax()
        rows.append(
            {
                "onset": idx.min() - pd.Timedelta(window),
                "peak": peak,
                "role": role,
                "p_before": lagged.loc[peak, role],
                "p_after": probs.loc[peak, role],
                "delta": delta.loc[peak, role],
                "magnitude": biggest.loc[peak],
            }
        )
    return pd.DataFrame(rows)
