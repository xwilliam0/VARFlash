"""Run the causal visual goal detector over a whole video, as it would run live.

Frames are sampled at 12.5 fps by timestamp; the CNN runs once per frame and the GRU over
the last ``frames`` frames for each time, so the score at t only uses video up to t.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from ..cards.clips import NEAREST_TOL_S
from ..cards.train import FRAME_SHAPE
from .clips import CLIP_FPS

log = logging.getLogger(__name__)


def iter_frames(video: Path, fps: float = CLIP_FPS):
    """Yield (sample time, RGB frame resized to FRAME_SHAPE) every 1/fps s of ``video``."""
    cap = cv2.VideoCapture(str(video))
    k = 0
    while cap.grab():
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 + NEAREST_TOL_S
        if k / fps > t:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        if frame.shape[:2] != FRAME_SHAPE:
            frame = cv2.resize(frame, FRAME_SHAPE[::-1], interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(frame[..., ::-1])
        while k / fps <= t:
            yield k / fps, rgb
            k += 1
    cap.release()


@torch.no_grad()
def video_goal_scores(model, ckpt: dict, video: Path, batch: int = 256, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """(times, live P(goal by t)) at 12.5 fps over ``video``; 0 until a full context window exists."""
    device = next(model.parameters()).device
    frames = ckpt["config"]["frames"]
    model.eval()
    feats, times, buf = [], [], []

    def flush():
        x = torch.from_numpy(np.stack(buf)).to(device).permute(0, 3, 1, 2).float()
        with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
            x = (x - model.mean) / model.std
            f = model.proj(model.backbone(x.contiguous(memory_format=torch.channels_last)))
        feats.append(f.float())
        buf.clear()

    for t, rgb in iter_frames(video):
        times.append(t)
        buf.append(rgb)
        if len(buf) == batch:
            flush()
    if buf:
        flush()
    if not feats:
        return np.zeros(0), np.zeros(0)
    f = torch.cat(feats)
    p = np.zeros(len(f), np.float32)
    for s in range(frames - 1, len(f), chunk):
        e = min(s + chunk, len(f))
        win = f[s - frames + 1 : e].unfold(0, frames, 1).permute(0, 2, 1)  # (e - s, frames, D)
        with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
            h, _ = model.gru(win)
            p[s:e] = torch.sigmoid(model.head(h[:, -1]).float())[:, 0].cpu().numpy()
    log.info("visual goal scores: %d frames (%.1f min) of %s", len(p), len(p) / CLIP_FPS / 60, Path(video).name)
    return np.array(times), p


def to_grid(times: np.ndarray, p: np.ndarray, fps: float, n: int) -> np.ndarray:
    """Max of the live score over each (j/fps - 1/fps, j/fps] bin, so grid value j is known at j/fps."""
    out = np.zeros(n, np.float32)
    j = np.clip(np.ceil(times * fps - 1e-9).astype(int), 0, n - 1)
    np.maximum.at(out, j, p)
    return out


def goal_alerts(score: np.ndarray, fps: float, fire_at: float = 0.9, merge_s: float = 60.0,
                lag_s: float = 1.0) -> pd.DataFrame:
    """An alert each time the live score crosses ``fire_at``; crossings within ``merge_s`` of the
    previous one belong to the same alert (the replays that follow a goal). The incident is put
    ``lag_s`` before the crossing, the typical rise time after the ball crosses the line.

    On full matches the score crosses 0.5 dozens of times (close-ups, replays, scrambles it never
    saw in shot-clip training), so the default threshold is high."""
    on = np.flatnonzero(score >= fire_at)
    rows = []
    if len(on):
        for seg in np.split(on, np.flatnonzero(np.diff(on) > merge_s * fps) + 1):
            fire, peak = seg[0], float(score[seg].max())
            rows.append({"frame": fire, "time_s": max(fire / fps - lag_s, 0.0), "fire_s": fire / fps, "label": "Goal",
                         "confidence": peak, "raw_score": peak})
    return pd.DataFrame(rows, columns=["frame", "time_s", "fire_s", "label", "confidence", "raw_score"])


def run_goal_vision(path: Path, video: Path, n_frames: int, fps: float, device: str):
    """Scores / alerts / meta in the same shape as ``replay.build.run_model``."""
    from .train import load_goal_model

    model, ckpt = load_goal_model(path, device)
    times, p = video_goal_scores(model, ckpt, video)
    s = to_grid(times, p, fps, n_frames)
    meta = {"path": str(path), "causal": True, "delay_s": 0.0, "classes": ["Goal"], "kind": "vision",
            "context_s": round(ckpt["config"]["frames"] / CLIP_FPS, 2)}
    return {"Goal": s}, goal_alerts(s, fps), meta


def is_goal_vision(path: Path) -> bool:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return "gap_s" in ckpt and "state_dict" in ckpt and math.isfinite(ckpt["gap_s"])
