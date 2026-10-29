"""Raw-frame clips around candidate card moments in SoccerNet broadcasts.

Each clip is a 24 s window (4 fps, native 224p) centred on a moment of interest:
a real card, a foul that drew no card, a random moment, or a candidate peak from the
feature-based spotter. Frames are stored as JPEG bytes (~1.5 MB per clip) so training
can re-read them quickly without seeking through full match videos.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..paths import SOCCERNET_DATA
from ..spotting.data import FPS, available_games, half_events

log = logging.getLogger(__name__)

CLIP_FPS = 4
STORE_HALF_S = 12.0  # stored window: centre +- 12 s
CLIP_FRAMES = int(2 * STORE_HALF_S * CLIP_FPS)  # 96
VIDEO_FILE = "{half}_224p.mkv"
CLIPS_DIR = SOCCERNET_DATA / "card_clips"
CARD_CLASSES = ["Yellow card", "Sending off"]
_SOURCE = {"Yellow card": 0, "Red card": 1, "Yellow->red card": 1}


def card_events(game_dir: Path) -> dict[int, list[tuple[float, int]]]:
    """{half: [(time_s, card_idx)]}, card_idx 0 = yellow, 1 = sending off."""
    labels = json.loads((game_dir / "Labels-v2.json").read_text())
    out: dict[int, list[tuple[float, int]]] = {1: [], 2: []}
    for a in labels["annotations"]:
        if a["label"] in _SOURCE:
            out[int(a["gameTime"][0])].append((int(a["position"]) / 1000, _SOURCE[a["label"]]))
    return out


def read_windows(video: Path, centers_s: list[float], half_s: float = STORE_HALF_S, fps: int = CLIP_FPS,
                 jpeg_quality: int = 90) -> list[list[bytes]]:
    """One sequential pass over ``video``; returns, per centre, JPEG frames at ``fps``.

    Frames outside the video are black. Sequential ``grab()`` is exact and, for many
    windows per half, faster than seeking (which lands on keyframes in mkv).
    """
    cap = cv2.VideoCapture(str(video))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = int(2 * half_s * fps)
    wanted: dict[int, list[tuple[int, int]]] = {}
    for c, t0 in enumerate(centers_s):
        for k in range(n):
            f = int(round((t0 - half_s + k / fps) * vfps))
            wanted.setdefault(f, []).append((c, k))
    out: list[list[bytes | None]] = [[None] * n for _ in centers_s]
    last = max((f for f in wanted if f < n_frames), default=-1)
    shape = None
    for f in range(last + 1):
        if not cap.grab():
            break
        if f in wanted:
            ok, frame = cap.retrieve()
            if not ok:
                continue
            shape = frame.shape
            buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])[1].tobytes()
            for c, k in wanted[f]:
                out[c][k] = buf
    cap.release()
    black = cv2.imencode(".jpg", np.zeros(shape or (224, 398, 3), np.uint8))[1].tobytes()
    return [[b if b is not None else black for b in clip] for clip in out]


def clip_id(game: str, half: int, center_s: float) -> str:
    g = hashlib.md5(Path(game).relative_to(SOCCERNET_DATA).as_posix().encode()).hexdigest()[:12]
    return f"{g}_{half}_{int(round(center_s * CLIP_FPS)):06d}"


def _extract_half(game: str, half: int, rows: list[dict], out_dir: Path) -> list[dict]:
    todo = [r for r in rows if not (out_dir / f"{r['clip_id']}.pkl").exists()]
    if todo:
        clips = read_windows(Path(game) / VIDEO_FILE.format(half=half), [r["center_s"] for r in todo])
        for r, frames in zip(todo, clips):
            (out_dir / f"{r['clip_id']}.pkl").write_bytes(pickle.dumps(frames))
    return rows


def _events_in_window(events: list[tuple[float, int]], center: float) -> str:
    return json.dumps([(round(t - center, 2), c) for t, c in events if abs(t - center) <= STORE_HALF_S])


def plan_training_clips(split: str, foul_neg_per_half: int = 4, random_neg_per_half: int = 2,
                        candidates: pd.DataFrame | None = None, seed: int = 0) -> pd.DataFrame:
    """Card moments + fouls without a card (hard negatives) + random moments (+ spotter candidates)."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in available_games(split):
        if not all((g / VIDEO_FILE.format(half=h)).exists() for h in (1, 2)):
            continue
        cards = card_events(g)
        fouls = half_events(g, {"Foul": ["Foul"]})
        for h in (1, 2):
            ev = cards[h]
            n_sec = len(np.load(g / f"{h}_ResNET_TF2_PCA512.npy", mmap_mode="r")) / FPS
            centers = [(t, "card") for t, _ in ev]
            far = [f / FPS for f, _ in fouls[h] if all(abs(f / FPS - t) > 30 for t, _ in ev)]
            centers += [(float(t), "foul") for t in rng.permutation(far)[:foul_neg_per_half]]
            centers += [(float(t), "random") for t in rng.uniform(10, max(n_sec - 10, 11), random_neg_per_half)]
            if candidates is not None:
                c = candidates[(candidates["game"] == str(g)) & (candidates["half"] == h)]
                centers += [(float(t), "candidate") for t in c["center_s"]]
            for t, kind in centers:
                rows.append({"clip_id": clip_id(str(g), h, t), "split": split, "game": str(g), "half": h,
                             "center_s": t, "kind": kind, "events": _events_in_window(ev, t)})
    return pd.DataFrame(rows).drop_duplicates("clip_id").reset_index(drop=True)


def extract(plan: pd.DataFrame, workers: int = 12, out_dir: Path = CLIPS_DIR) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = list(plan.groupby(["game", "half"]))
    with ProcessPoolExecutor(workers) as pool:
        futs = [pool.submit(_extract_half, g, h, df.to_dict("records"), out_dir) for (g, h), df in groups]
        for i, f in enumerate(as_completed(futs), 1):
            f.result()
            if i % 50 == 0 or i == len(futs):
                log.info("extracted clips for %d/%d halves", i, len(futs))
    return plan


def load_clip(cid: str, out_dir: Path = CLIPS_DIR) -> list[bytes]:
    return pickle.loads((out_dir / f"{cid}.pkl").read_bytes())
