"""Run a trained spotter over a full half and turn frame scores into discrete events."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import maximum_filter1d

from .data import CLASSES, FPS
from .model import TemporalSpotter


def load_spotter(path: Path | str, device: str = "cpu") -> tuple[TemporalSpotter, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TemporalSpotter(**ckpt["model_args"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    ckpt.setdefault("classes", CLASSES)
    ckpt.setdefault("delay", 0)
    return model, ckpt


def _device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def frame_scores(model: TemporalSpotter, features: np.ndarray, window: int = 120) -> np.ndarray:
    """Sigmoid scores (N, C).

    Bidirectional models: average of overlapping half-stride windows (each frame sees
    up to ``window`` frames of future). Causal models: each frame is scored by a window
    that ends at or after it but whose output for that frame only depends on the past,
    and has at least ``window // 2`` frames of history, so the scores are live-valid.
    """
    n = len(features)
    feats = torch.from_numpy(np.array(features, np.float32)).to(_device(model))
    starts = list(range(0, max(n - window, 0) + 1, window // 2))
    if starts[-1] + window < n:
        starts.append(n - window)
    batch = torch.stack([feats[s : s + window] for s in starts]) if n >= window else feats[None]
    out = torch.cat([torch.sigmoid(model(b)) for b in batch.split(256)]).float().cpu().numpy()
    n_classes = out.shape[-1]
    if getattr(model, "causal", False):
        scores = np.zeros((n, n_classes), np.float32)
        half = window // 2
        for k, s in enumerate(starts):
            keep = 0 if s == 0 else half  # first window: no older history exists anyway
            scores[s + keep : s + out.shape[1]] = out[k, keep:]
        return scores
    acc = np.zeros((n, n_classes), np.float32)
    cnt = np.zeros((n, 1), np.float32)
    for k, s in enumerate(starts):
        acc[s : s + out.shape[1]] += out[k]
        cnt[s : s + out.shape[1]] += 1
    return acc / np.maximum(cnt, 1)


def event_scores(scores: np.ndarray, delay: int) -> np.ndarray:
    """Undo a causal model's target delay: row t then scores an event *at* frame t."""
    if delay <= 0:
        return scores
    out = np.zeros_like(scores)
    out[: len(scores) - delay] = scores[delay:]
    return out


def calibrate(confidence: np.ndarray, calibrator: dict | None) -> np.ndarray:
    """Isotonic map (fitted on validation peaks) from raw score to P(real event)."""
    if not calibrator:
        return confidence
    return np.interp(confidence, calibrator["x"], calibrator["y"])


def peak_pick(scores: np.ndarray, classes: list[str] = CLASSES, nms_window_s: float = 10.0,
              min_score: float = 0.05, calibrators: dict | None = None) -> pd.DataFrame:
    """Local maxima per class within ``nms_window_s`` -> rows (frame, time_s, label, confidence)."""
    size = max(int(nms_window_s * FPS), 1) | 1
    rows = []
    for c, name in enumerate(classes):
        s = scores[:, c]
        peaks = np.flatnonzero((s >= min_score) & (s == maximum_filter1d(s, size, mode="constant")))
        if len(peaks) > 1:  # flat plateaus give several equal maxima; keep the first of each run
            peaks = peaks[np.concatenate([[True], np.diff(peaks) > size // 2])]
        conf = calibrate(s[peaks], (calibrators or {}).get(name))
        rows += [{"frame": int(f), "time_s": f / FPS, "label": name, "confidence": float(p), "raw_score": float(r)}
                 for f, p, r in zip(peaks, conf, s[peaks])]
    return pd.DataFrame(rows, columns=["frame", "time_s", "label", "confidence", "raw_score"])


def spot(model: TemporalSpotter, features: np.ndarray, ckpt: dict | None = None, **kw) -> pd.DataFrame:
    """Discrete events for one half. With a causal checkpoint, ``fire_s`` is when a live
    system would have known (event time + look-ahead delay)."""
    ckpt = ckpt or {}
    delay = ckpt.get("delay", 0)
    kw.setdefault("classes", ckpt.get("classes", CLASSES))
    kw.setdefault("calibrators", ckpt.get("calibrators"))
    window = ckpt.get("config", {}).get("window", 120)
    ev = peak_pick(event_scores(frame_scores(model, features, window), delay), **kw)
    ev["fire_s"] = ev["time_s"] + delay / FPS
    return ev.sort_values("time_s").reset_index(drop=True)
