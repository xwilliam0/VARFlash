"""Match-level bootstrap confidence intervals for spotting AP, and paired model differences.

Rare events make test AP noisy (about 21 sending-offs in the test split), so two models can
differ by chance. Resampling whole matches with replacement keeps each event together with
the predictions around it. Predictions are matched to events within one half, so a
prediction's true/false flag never changes under resampling: a resample only reweights
matches, which makes thousands of resamples cheap. With every weight 1 the AP equals
``metrics.average_precision`` exactly.

Comparing models on the *same* resamples (paired) cancels most of the match-to-match noise,
so a difference can be clearly real even when the two intervals overlap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import match_predictions

TOLERANCES = (5, 10, 30, 60)


def game_of(video: str) -> str:
    """``"<game dir>#<half>"`` -> ``"<game dir>"``."""
    return video.rsplit("#", 1)[0]


def prepare(preds: pd.DataFrame, gts: pd.DataFrame, classes: list[str], games: list[str],
            tolerances=TOLERANCES) -> dict:
    """{(class, tol): (game index per prediction, hit per prediction, events per game)}, confidence-sorted."""
    gidx = {g: i for i, g in enumerate(games)}

    def index(videos: pd.Series) -> np.ndarray:
        idx = videos.map(game_of).map(gidx)
        if idx.isna().any():
            raise ValueError(f"games not in the test list, e.g. {game_of(videos[idx.isna()].iloc[0])!r}")
        return idx.to_numpy(int)

    out = {}
    for c in classes:
        p = preds[preds["label"] == c].reset_index(drop=True)
        g = gts[gts["label"] == c]
        n_gt = np.bincount(index(g["video"]), minlength=len(games))
        order = np.argsort(-p["confidence"].to_numpy(), kind="stable")
        pg = index(p["video"])[order]
        for t in tolerances:
            out[(c, t)] = (pg, match_predictions(p, g, t)[order], n_gt)
    return out


def weighted_ap(pred_game: np.ndarray, hit: np.ndarray, n_gt: np.ndarray, w: np.ndarray) -> float:
    """All-point interpolated AP when match ``i`` appears ``w[i]`` times."""
    n = float((w * n_gt).sum())
    if n == 0:
        return float("nan")
    wp = w[pred_game]
    keep = wp > 0
    if not keep.any():
        return 0.0
    wp, h = wp[keep], hit[keep]
    cum_tp = np.cumsum(wp * h)
    precision = cum_tp / np.cumsum(wp)
    recall = cum_tp / n
    mprec = np.maximum.accumulate(np.r_[0.0, precision, 0.0][::-1])[::-1]
    mrec = np.r_[0.0, recall, recall[-1]]
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(((mrec[idx + 1] - mrec[idx]) * mprec[idx + 1]).sum())


def resample_weights(n_games: int, n_boot: int, seed: int = 0) -> np.ndarray:
    """(1 + n_boot, n_games) match counts; row 0 is the original test set."""
    rng = np.random.default_rng(seed)
    return np.vstack([np.ones(n_games), rng.multinomial(n_games, np.full(n_games, 1 / n_games), size=n_boot)])


def ap_samples(pred_tables: list[pd.DataFrame], gts: pd.DataFrame, classes: list[str], games: list[str],
               weights: np.ndarray, tolerances=TOLERANCES) -> np.ndarray:
    """(1 + n_boot, n_classes) AP averaged over tolerances and over ``pred_tables`` (e.g. seeds)."""
    out = np.zeros((len(weights), len(classes)))
    for preds in pred_tables:
        prep = prepare(preds, gts, classes, games, tolerances)
        for k, c in enumerate(classes):
            per_tol = np.array([[weighted_ap(*prep[(c, t)], w) for w in weights] for t in tolerances])
            out[:, k] += per_tol.mean(0)
    return out / len(pred_tables)


def _summary(samples: np.ndarray) -> dict:
    boot = samples[1:]
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {"value": float(samples[0]), "ci_low": float(lo), "ci_high": float(hi)}


def compare(pred_sets: dict[str, list[pd.DataFrame]], gts: pd.DataFrame, classes: list[str], games: list[str],
            baseline: str, n_boot: int = 1000, seed: int = 0,
            tolerances=TOLERANCES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """AP with 95% CI per model and class, and each model minus ``baseline`` on the same resamples.

    Class ``"mean"`` averages ``classes``. ``p_not_better`` is the share of resamples where the
    model does not beat the baseline: below ~0.05 the gain is unlikely to be match noise.
    """
    w = resample_weights(len(games), n_boot, seed)
    samples = {}
    for name, tables in pred_sets.items():
        s = ap_samples(tables, gts, classes, games, w, tolerances)
        samples[name] = np.column_stack([s, np.nanmean(s, axis=1)])
    names = [*classes, "mean"]
    aps = pd.DataFrame([{"model": m, "class": c, "n_models": len(pred_sets[m]), **_summary(s[:, k])}
                        for m, s in samples.items() for k, c in enumerate(names)])
    diffs = []
    for m, s in samples.items():
        if m == baseline:
            continue
        d = s - samples[baseline]
        for k, c in enumerate(names):
            diffs.append({"model": m, "baseline": baseline, "class": c, **_summary(d[:, k]),
                          "p_not_better": float(np.mean(d[1:, k] <= 0))})
    return aps, pd.DataFrame(diffs)
