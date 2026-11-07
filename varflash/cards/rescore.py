"""Two-stage card detection: feature spotter proposes moments, the visual model verifies them.

The spotter (2 fps frozen features) is cheap and runs on the whole match; the visual
model (raw frames, 4 fps) only looks at a 12 s clip around each candidate. A logistic
regression fitted on the validation split fuses the two scores.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..spotting.data import FPS, available_games, load_halves
from ..spotting.infer import event_scores, frame_scores, load_spotter, peak_pick
from ..spotting.metrics import calibration_report, match_predictions, spotting_report
from .clips import CARD_CLASSES, card_events, clip_id

TOLERANCES = (5, 10, 30, 60)


def spotter_candidates(spotter_path, split: str, min_score: float = 0.05, merge_s: float = 4.0,
                       device: str = "cuda") -> pd.DataFrame:
    """Card candidate moments from the feature spotter, with its score for each card class."""
    model, ck = load_spotter(spotter_path, device)
    classes, delay = ck["classes"], ck.get("delay", 0)
    idx = [classes.index(c) for c in CARD_CLASSES]
    window = ck.get("config", {}).get("window", 120)
    rows = []
    for h in load_halves(available_games(split), radius=0, classes=ck["config"]["classes"]):
        s = event_scores(frame_scores(model, h.features, window), delay)[:, idx]
        peaks = peak_pick(s, CARD_CLASSES, min_score=min_score).sort_values("time_s")
        clusters: list[list] = []
        for p in peaks.itertuples():
            if clusters and p.time_s - clusters[-1][-1].time_s <= merge_s:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        for cl in clusters:
            t = max(cl, key=lambda p: p.confidence).time_s
            f = int(t * FPS)
            sc = s[max(f - 4, 0) : f + 5].max(0)
            rows.append({"game": h.game, "half": h.half, "center_s": float(t), "clip_id": clip_id(h.game, h.half, t),
                         "sp_yellow": float(sc[0]), "sp_red": float(sc[1])})
    return pd.DataFrame(rows)


def _game_key(game: str) -> str:
    """Game key independent of the home-dir alias the path was saved under (/home/... vs /orcd/home/...)."""
    return str(game).split("/soccernet/")[-1]


def spotter_at_candidates(spotter_path, split: str, cands: pd.DataFrame, radius: int = 4,
                          device: str = "cpu") -> pd.DataFrame:
    """Max score of every spotter class within ``radius`` frames of each candidate, indexed like ``cands``.

    Lets a newer spotter score the existing candidates (and their clips) without re-extracting video.
    """
    model, ck = load_spotter(spotter_path, device)
    classes, window = ck["classes"], ck.get("config", {}).get("window", 120)
    by_half = {(_game_key(g), h): d for (g, h), d in cands.groupby(["game", "half"])}
    rows = []
    for h in load_halves(available_games(split), radius=0, classes=ck["config"]["classes"]):
        c = by_half.get((_game_key(h.game), h.half))
        if c is None:
            continue
        s = event_scores(frame_scores(model, h.features, window), ck.get("delay", 0))
        for i, t in zip(c.index, c["center_s"]):
            f = int(t * FPS)
            rows.append({"row": i, **dict(zip(classes, s[max(f - radius, 0) : f + radius + 1].max(0)))})
    out = pd.DataFrame(rows).set_index("row").reindex(cands.index)
    if out.isna().any().any():
        raise ValueError(f"{int(out.isna().any(axis=1).sum())} candidates have no features in split {split!r}")
    return out


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _features(df: pd.DataFrame) -> np.ndarray:
    return np.stack([_logit(df[c].to_numpy()) for c in ("sp_yellow", "sp_red", "pv_yellow", "pv_red")], 1)


def _truth(df: pd.DataFrame, tol: float = 10.0) -> np.ndarray:
    """(N, 2): is there a real card of each class within ``tol`` s of the candidate."""
    ev = {}
    y = np.zeros((len(df), len(CARD_CLASSES)), int)
    for i, r in enumerate(df.itertuples()):
        if r.game not in ev:
            ev[r.game] = card_events(Path(r.game))
        for t, c in ev[r.game][r.half]:
            if abs(t - r.center_s) <= tol:
                y[i, c] = 1
    return y


def fit_fusion(valid: pd.DataFrame) -> list[LogisticRegression]:
    X, Y = _features(valid), _truth(valid)
    return [LogisticRegression(C=1.0, max_iter=1000).fit(X, Y[:, k]) for k in range(len(CARD_CLASSES))]


def _gts(split: str) -> pd.DataFrame:
    rows = []
    for g in available_games(split):
        for h, evs in card_events(g).items():
            rows += [{"video": f"{g}#{h}", "time_s": t, "label": CARD_CLASSES[c]} for t, c in evs]
    return pd.DataFrame(rows)


def _fused_logit(df: pd.DataFrame, fusion: list[LogisticRegression]) -> np.ndarray:
    return np.stack([m.decision_function(_features(df)) for m in fusion], 1)


def _fused_raw(df: pd.DataFrame, fusion: list[LogisticRegression]) -> np.ndarray:
    return 1 / (1 + np.exp(-_fused_logit(df, fusion)))


def _new_card(df: pd.DataFrame, p: np.ndarray, split: str, tol: float = 10.0) -> np.ndarray:
    """(N, 2): candidate is the best-scored match of a not-yet-claimed real card (one alert per card)."""
    gts = _gts(split)
    gts["video"] = gts["video"].map(_game_key)
    video = df["game"].map(_game_key) + "#" + df["half"].astype(str)
    return np.stack([match_predictions(pd.DataFrame({"video": video, "time_s": df["center_s"], "confidence": p[:, k]}),
                                       gts[gts["label"] == c], tol) for k, c in enumerate(CARD_CLASSES)], 1)


def fit_card_calibration(valid: pd.DataFrame, fusion: list[LogisticRegression], split: str = "valid") -> np.ndarray:
    """(2, 2) Platt (slope, intercept) per class on the fused logit -> P(new real card).

    The fusion is fitted to "any card within 10 s", which counts several candidates around one
    card as hits; a one-to-one matched target is what an alert's stake should be sized on.
    Platt rather than isotonic: valid has only ~25 sending-off candidates.
    """
    z = _fused_logit(valid, fusion)
    hit = _new_card(valid, 1 / (1 + np.exp(-z)), split)
    out = np.zeros((len(CARD_CLASSES), 2))
    for k in range(len(CARD_CLASSES)):
        m = LogisticRegression(C=1e4, max_iter=1000).fit(z[:, k : k + 1], hit[:, k])
        out[k] = m.coef_[0, 0], m.intercept_[0]
    return out


def card_probability(df: pd.DataFrame, fusion: list[LogisticRegression], platt: np.ndarray | None = None) -> np.ndarray:
    """(N, 2) fused [yellow, red] probability per candidate, Platt-calibrated when ``platt`` is given."""
    z = _fused_logit(df, fusion)
    if platt is not None:
        z = z * platt[:, 0] + platt[:, 1]
    return 1 / (1 + np.exp(-z))


def calibration_table(test: pd.DataFrame, fusion: list[LogisticRegression], platt: np.ndarray,
                      split: str = "test") -> pd.DataFrame:
    """Brier / ECE of raw vs calibrated fused probabilities against one-alert-per-card truth."""
    raw, cal = card_probability(test, fusion), card_probability(test, fusion, platt)
    rows = []
    for which, p in (("raw", raw), ("calibrated", cal)):
        hit = _new_card(test, p, split)
        for k, c in enumerate(CARD_CLASSES):
            hi = p[:, k] >= 0.1
            rows.append({"class": c, "which": which, **calibration_report(p[:, k], hit[:, k]),
                         "n_p>=0.1": int(hi.sum()), "mean_p>=0.1": float(p[hi, k].mean()) if hi.any() else np.nan,
                         "hit_rate_p>=0.1": float(hit[hi, k].mean()) if hi.any() else np.nan})
    return pd.DataFrame(rows)


def compare(test: pd.DataFrame, fusion: list[LogisticRegression], split: str = "test") -> pd.DataFrame:
    """AP per card class for spotter-only, visual-only and fused confidences on the same candidates."""
    gts = _gts(split)
    gts["video"] = gts["video"].map(_game_key)
    video = test["game"].map(_game_key) + "#" + test["half"].astype(str)
    fused = _fused_raw(test, fusion)
    scores = {"spotter": test[["sp_yellow", "sp_red"]].to_numpy(), "visual": test[["pv_yellow", "pv_red"]].to_numpy(),
              "fused": fused}
    rows = []
    for name, s in scores.items():
        preds = pd.concat([pd.DataFrame({"video": video, "time_s": test["center_s"], "label": c, "confidence": s[:, k]})
                           for k, c in enumerate(CARD_CLASSES)], ignore_index=True)
        rep = spotting_report(preds, gts, CARD_CLASSES, TOLERANCES)
        rep.insert(0, "scores", name)
        rep["mean_AP"] = rep[[f"AP@{t}s" for t in TOLERANCES]].mean(1)
        rows.append(rep)
    return pd.concat(rows, ignore_index=True)
