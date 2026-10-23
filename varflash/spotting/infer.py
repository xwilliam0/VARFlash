"""Run a trained spotter over a full half and turn frame scores into discrete events."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import CLASSES, FPS
from .model import TemporalSpotter


def load_spotter(path: Path | str) -> tuple[TemporalSpotter, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TemporalSpotter(**ckpt["model_args"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def frame_scores(model: TemporalSpotter, features: np.ndarray, window: int = 120) -> np.ndarray:
    """Sigmoid scores (N, C), averaging overlapping half-stride windows."""
    n = len(features)
    feats = torch.from_numpy(np.array(features, np.float32))
    acc = np.zeros((n, len(CLASSES)), np.float32)
    cnt = np.zeros((n, 1), np.float32)
    starts = list(range(0, max(n - window, 0) + 1, window // 2))
    if starts[-1] + window < n:
        starts.append(n - window)
    for s in starts:
        chunk = feats[s : s + window].unsqueeze(0)
        acc[s : s + chunk.shape[1]] += torch.sigmoid(model(chunk))[0].numpy()
        cnt[s : s + chunk.shape[1]] += 1
    return acc / np.maximum(cnt, 1)


def peak_pick(scores: np.ndarray, nms_window_s: float = 10.0, min_score: float = 0.05) -> pd.DataFrame:
    """Greedy temporal NMS per class -> rows (frame, time_s, label, confidence)."""
    half_w = int(nms_window_s * FPS / 2)
    rows = []
    for c, name in enumerate(CLASSES):
        s = scores[:, c].copy()
        while True:
            f = int(np.argmax(s))
            if s[f] < min_score:
                break
            rows.append({"frame": f, "time_s": f / FPS, "label": name, "confidence": float(s[f])})
            s[max(f - half_w, 0) : f + half_w + 1] = 0
    return pd.DataFrame(rows, columns=["frame", "time_s", "label", "confidence"])


def spot(model: TemporalSpotter, features: np.ndarray, **kw) -> pd.DataFrame:
    return peak_pick(frame_scores(model, features), **kw).sort_values("time_s").reset_index(drop=True)
