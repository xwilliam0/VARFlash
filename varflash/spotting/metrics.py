"""Tolerance-based average precision for action spotting (SoccerNet-style), plus
calibration and live-latency metrics for the card detector."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FPS


def match_predictions(preds: pd.DataFrame, gts: pd.DataFrame, tolerance_s: float) -> np.ndarray:
    """True-positive flag per prediction (in ``preds`` order), greedy by confidence.

    A prediction is a true positive if it lies within ``tolerance_s`` of a
    not-yet-matched ground-truth event in the same video (half).
    """
    gt_by_video = {v: g["time_s"].to_numpy() for v, g in gts.groupby("video")}
    used = {v: np.zeros(len(t), bool) for v, t in gt_by_video.items()}
    hits = np.zeros(len(preds), bool)
    order = np.argsort(-preds["confidence"].to_numpy(), kind="stable")
    videos, times = preds["video"].to_numpy(), preds["time_s"].to_numpy()
    for k in order:
        t = gt_by_video.get(videos[k])
        if t is None:
            continue
        d = np.abs(t - times[k])
        d[used[videos[k]]] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tolerance_s:
            used[videos[k]][j] = True
            hits[k] = True
    return hits


def average_precision(preds: pd.DataFrame, gts: pd.DataFrame, tolerance_s: float) -> float:
    """AP for one class. ``preds``: (video, time_s, confidence); ``gts``: (video, time_s)."""
    n_gt = len(gts)
    if n_gt == 0:
        return float("nan")
    if preds.empty:
        return 0.0
    order = np.argsort(-preds["confidence"].to_numpy(), kind="stable")
    tp = match_predictions(preds, gts, tolerance_s)[order].astype(float)
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


# ---- calibration --------------------------------------------------------------


def fit_isotonic(confidence: np.ndarray, is_true: np.ndarray, n_points: int = 50) -> dict:
    """Monotone map raw score -> P(true event), stored as interpolation knots."""
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(confidence, is_true.astype(float))
    x = np.unique(np.quantile(confidence, np.linspace(0, 1, n_points)))
    return {"x": x.tolist(), "y": iso.predict(x).tolist()}


def calibration_report(confidence: np.ndarray, is_true: np.ndarray, bins: int = 10) -> dict:
    """Brier score and expected calibration error (equal-width bins)."""
    is_true = is_true.astype(float)
    if len(confidence) == 0:
        return {"n": 0, "brier": float("nan"), "ece": float("nan")}
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(confidence, edges) - 1, 0, bins - 1)
    ece = sum(abs(confidence[idx == b].mean() - is_true[idx == b].mean()) * (idx == b).mean()
              for b in range(bins) if (idx == b).any())
    return {"n": int(len(confidence)), "brier": float(np.mean((confidence - is_true) ** 2)), "ece": float(ece)}


# ---- live latency -------------------------------------------------------------


def first_crossings(scores: np.ndarray, threshold: float, refractory_s: float = 20.0) -> np.ndarray:
    """Frames where a live score first rises above ``threshold`` (then ignored for ``refractory_s``)."""
    above = np.flatnonzero(scores >= threshold)
    out, last = [], -np.inf
    for f in above:
        if (f == 0 or scores[f - 1] < threshold) and f - last >= refractory_s * FPS:
            out.append(f)
            last = f
    return np.array(out, int)


def latency_report(live: list[tuple[str, np.ndarray, np.ndarray]], threshold: float, max_delay_s: float = 120.0,
                   early_s: float = 10.0, budgets=(15, 30, 45, 60, 75)) -> dict:
    """Live alerting quality for one class.

    ``live``: per half, (video, live scores (N,), true event times in s). A live score at
    frame t only uses frames <= t, so a crossing at t is when the alert would fire. An
    alert counts for an event if it fires between ``early_s`` before and ``max_delay_s``
    after it. Returns precision, recall within each delay budget, and median delay.
    """
    delays, n_alerts, n_true_alerts, n_events = [], 0, 0, 0
    for _, scores, events in live:
        fires = first_crossings(scores, threshold) / FPS
        n_alerts += len(fires)
        n_events += len(events)
        used = np.zeros(len(fires), bool)
        for t in sorted(events):
            ok = np.flatnonzero(~used & (fires >= t - early_s) & (fires <= t + max_delay_s))
            if len(ok):
                used[ok[0]] = True
                delays.append(fires[ok[0]] - t)
        n_true_alerts += int(used.sum())
    d = np.array(delays)
    out = {"threshold": threshold, "events": n_events, "alerts": n_alerts,
           "precision": n_true_alerts / n_alerts if n_alerts else float("nan"),
           "median_delay_s": float(np.median(d)) if len(d) else float("nan")}
    for b in budgets:
        out[f"recall@{b}s"] = float((d <= b).sum() / n_events) if n_events else float("nan")
    return out


def pick_threshold(live, target_precision: float = 0.5, grid=None, **kw) -> float:
    """Lowest threshold whose live precision reaches ``target_precision`` (max recall)."""
    grid = np.linspace(0.05, 0.95, 37) if grid is None else grid
    best = float(grid[-1])
    for th in grid:
        r = latency_report(live, float(th), **kw)
        if r["alerts"] and r["precision"] >= target_precision:
            return float(th)
    return best
