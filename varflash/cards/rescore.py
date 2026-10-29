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
from ..spotting.metrics import spotting_report
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


def compare(test: pd.DataFrame, fusion: list[LogisticRegression], split: str = "test") -> pd.DataFrame:
    """AP per card class for spotter-only, visual-only and fused confidences on the same candidates."""
    gts = _gts(split)
    video = test["game"] + "#" + test["half"].astype(str)
    fused = np.stack([m.predict_proba(_features(test))[:, 1] for m in fusion], 1)
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
