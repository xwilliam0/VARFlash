"""Score a 17-class spotter with SoccerNet's official evaluator (average-mAP, loose and tight).

Numbers from here are directly comparable to the SoccerNet-v2 action-spotting
benchmark tables (e.g. NetVLAD++ / CALF on the same ResNet PCA-512 features).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from ..paths import OUTPUTS, SOCCERNET_DATA
from .data import CLASSES, FPS, available_games, load_halves
from .infer import event_scores, frame_scores, peak_pick

log = logging.getLogger(__name__)


def write_predictions(model, ckpt: dict, split: str = "test", out_dir: Path | None = None,
                      max_games: int | None = None) -> Path:
    """One ``results_spotting.json`` per game, in the SoccerNet layout."""
    classes = ckpt.get("classes", CLASSES)
    if sorted(classes) != sorted(CLASSES):
        raise ValueError("the official evaluator needs a model trained on all 17 SoccerNet classes")
    out_dir = out_dir or OUTPUTS / f"soccernet_preds_{split}"
    window = ckpt.get("config", {}).get("window", 120)
    games = available_games(split)[:max_games]
    for g in games:
        anns = []
        for h in load_halves([g], radius=0):
            # Low threshold: average-mAP integrates over the whole PR curve.
            p = peak_pick(event_scores(frame_scores(model, h.features, window), ckpt.get("delay", 0)), classes,
                          min_score=0.01)
            for r in p.itertuples():
                ms = int(r.frame / FPS * 1000)
                anns.append({"gameTime": f"{h.half} - {ms // 60000:02d}:{ms // 1000 % 60:02d}", "label": r.label,
                             "position": str(ms), "half": str(h.half), "confidence": str(r.confidence)})
        rel = g.relative_to(SOCCERNET_DATA)
        (out_dir / rel).mkdir(parents=True, exist_ok=True)
        (out_dir / rel / "results_spotting.json").write_text(json.dumps({"UrlLocal": str(rel), "predictions": anns}))
    log.info("wrote predictions for %d games -> %s", len(games), out_dir)
    return out_dir


def official_scores(pred_dir: Path, split: str = "test") -> pd.DataFrame:
    from SoccerNet.Evaluation.ActionSpotting import evaluate
    from SoccerNet.Evaluation.utils import INVERSE_EVENT_DICTIONARY_V2

    rows = {}
    for metric in ("loose", "tight"):
        r = evaluate(SoccerNet_path=str(SOCCERNET_DATA), Predictions_path=str(pred_dir), split=split, version=2,
                     prediction_file="results_spotting.json", metric=metric)
        rows[f"{metric}_all"] = {"overall": r["a_mAP"], **{INVERSE_EVENT_DICTIONARY_V2[i]: v
                                                             for i, v in enumerate(r["a_mAP_per_class"])}}
        rows[f"{metric}_visible"] = {"overall": r["a_mAP_visible"]}
    return pd.DataFrame(rows)
