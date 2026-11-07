"""Two-stage offside detection: the feature spotter proposes moments, the visual model verifies them.

AP is computed over whole matches against every annotated offside, so offsides the spotter
never proposes still count as misses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..cards.clips import clip_id
from ..spotting.data import available_games, load_halves
from ..spotting.infer import event_scores, frame_scores, load_spotter, peak_pick
from ..spotting.metrics import spotting_report
from .clips import half_labels, offsides_near

TOLERANCES = (5, 10, 30, 60)


def spotter_candidates(spotter_path, split: str, min_score: float = 0.1, device: str = "cuda") -> pd.DataFrame:
    """Offside peaks of the feature spotter (after its 10 s NMS) with their raw score."""
    model, ck = load_spotter(spotter_path, device)
    cfg = ck.get("config", {})
    c = ck["classes"].index("Offside")
    rows = []
    for h in load_halves(available_games(split), radius=0, classes=cfg.get("classes", "all")):
        s = event_scores(frame_scores(model, h.features, cfg.get("window", 120), cfg.get("edge", 0)), ck["delay"])
        lab = half_labels(Path(h.game))[h.half]
        for p in peak_pick(s[:, [c]], ["Offside"], min_score=min_score).itertuples():
            rows.append({"game": h.game, "half": h.half, "center_s": p.time_s,
                         "clip_id": clip_id(h.game, h.half, p.time_s), "sp_offside": p.confidence,
                         "offsides": offsides_near(lab, p.time_s)})
    return pd.DataFrame(rows)


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _features(df: pd.DataFrame) -> np.ndarray:
    a, b = _logit(df["sp_offside"]), _logit(df["pv_offside"])
    return np.stack([a, b, a * b], 1)


def _truth(df: pd.DataFrame, tol: float = 5.0) -> np.ndarray:
    return np.array([any(abs(o) <= tol for o in json.loads(x)) for x in df["offsides"]], int)


def fit_fusion(valid: pd.DataFrame) -> LogisticRegression:
    return LogisticRegression(C=1.0, max_iter=1000).fit(_features(valid), _truth(valid))


def offside_gts(split: str) -> pd.DataFrame:
    rows = []
    for g in available_games(split):
        for h, lab in half_labels(g).items():
            rows += [{"video": f"{g}#{h}", "time_s": t, "label": "Offside"} for t, l in lab if l == "Offside"]
    return pd.DataFrame(rows)


def compare(test: pd.DataFrame, fusion: LogisticRegression, split: str = "test") -> pd.DataFrame:
    """Offside AP over whole matches for spotter-only, visual-only and fused confidences."""
    gts = offside_gts(split)
    video = test["game"] + "#" + test["half"].astype(str)
    scores = {"spotter": test["sp_offside"].to_numpy(), "visual": test["pv_offside"].to_numpy(),
              "fused": fusion.predict_proba(_features(test))[:, 1]}
    rows = []
    for name, s in scores.items():
        preds = pd.DataFrame({"video": video, "time_s": test["center_s"], "label": "Offside", "confidence": s})
        rep = spotting_report(preds, gts, ["Offside"], TOLERANCES)
        rep.insert(0, "scores", name)
        rep["mean_AP"] = rep[[f"AP@{t}s" for t in TOLERANCES]].mean(1)
        rows.append(rep)
    return pd.concat(rows, ignore_index=True)
