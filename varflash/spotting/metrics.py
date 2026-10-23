"""Tolerance-based average precision for action spotting (SoccerNet-style)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def average_precision(preds: pd.DataFrame, gts: pd.DataFrame, tolerance_s: float) -> float:
    """AP for one class.

    ``preds``: columns (video, time_s, confidence); ``gts``: columns (video, time_s).
    A prediction is a true positive if it lies within ``tolerance_s`` of a
    not-yet-matched ground-truth event in the same video (half).
    """
    n_gt = len(gts)
    if n_gt == 0:
        return float("nan")
    if preds.empty:
        return 0.0
    gt_by_video = {v: g["time_s"].to_numpy() for v, g in gts.groupby("video")}
    used = {v: np.zeros(len(t), bool) for v, t in gt_by_video.items()}
    tp = []
    for p in preds.sort_values("confidence", ascending=False).itertuples():
        times = gt_by_video.get(p.video)
        hit = False
        if times is not None:
            d = np.abs(times - p.time_s)
            d[used[p.video]] = np.inf
            j = int(np.argmin(d))
            if d[j] <= tolerance_s:
                used[p.video][j] = True
                hit = True
        tp.append(hit)
    tp = np.array(tp, float)
    cum_tp = np.cumsum(tp)
    precision = cum_tp / np.arange(1, len(tp) + 1)
    recall = cum_tp / n_gt
    # All-point interpolated area under the PR curve.
    mprec = np.concatenate([[0.0], precision, [0.0]])
    mrec = np.concatenate([[0.0], recall, [recall[-1]]])
    for i in range(len(mprec) - 2, -1, -1):
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mprec[idx + 1]).sum())


def spotting_report(preds: pd.DataFrame, gts: pd.DataFrame, classes: list[str], tolerances=(5, 10, 30, 60)) -> pd.DataFrame:
    rows = []
    for c in classes:
        p, g = preds[preds["label"] == c], gts[gts["label"] == c]
        row = {"class": c, "n_events": len(g)}
        for t in tolerances:
            row[f"AP@{t}s"] = average_precision(p, g, t)
        rows.append(row)
    return pd.DataFrame(rows)
