"""Raw-frame clips around candidate offside moments in SoccerNet broadcasts.

The feature spotter finds offsides easily but confuses them with every other stoppage
(ball out of play, fouls, missed shots). What tells them apart is mostly visual and small:
the assistant referee's raised flag near the touchline (cut off by the centre crop the
SoccerNet features use), players stopping and turning, and the defending side placing
the ball for the free kick. Clips keep the full 16:9 frame, from 10 s before to 14 s
after the moment, at 4 fps, stored as JPEG bytes.
"""

from __future__ import annotations

import json
import logging
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from ..cards.clips import CLIP_FPS, VIDEO_FILE, clip_id, read_windows
from ..paths import SOCCERNET_DATA
from ..spotting.data import FPS, LABEL_FILE, available_games

log = logging.getLogger(__name__)

PRE_S, POST_S = 10.0, 14.0
CLIP_FRAMES = int((PRE_S + POST_S) * CLIP_FPS)  # 96
CLIPS_DIR = SOCCERNET_DATA / "offside_clips"
# Stoppages the feature spotter mistakes for offsides; their moments are the hard negatives.
CONFUSERS = ("Ball out of play", "Foul", "Shots off target", "Shots on target", "Clearance")


def half_labels(game_dir: Path) -> dict[int, list[tuple[float, str]]]:
    labels = json.loads((game_dir / LABEL_FILE).read_text())
    out: dict[int, list[tuple[float, str]]] = {1: [], 2: []}
    for a in labels["annotations"]:
        out[int(a["gameTime"].split(" - ")[0])].append((int(a["position"]) / 1000, a["label"]))
    return out


def offsides_near(labels: list[tuple[float, str]], t: float) -> str:
    """JSON list of offside offsets (s) from ``t`` that fall inside the stored clip."""
    return json.dumps([round(x - t, 2) for x, l in labels if l == "Offside" and -PRE_S <= x - t <= POST_S])


def plan_clips(split: str, candidates: pd.DataFrame | None = None, confuser_per_half: int = 6,
               random_per_half: int = 2, seed: int = 0) -> pd.DataFrame:
    """Offsides + confusing stoppages + random moments (train/valid) and spotter candidates."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in available_games(split):
        if not all((g / VIDEO_FILE.format(half=h)).exists() for h in (1, 2)):
            continue
        labels = half_labels(g)
        for h in (1, 2):
            lab = labels[h]
            off = [t for t, l in lab if l == "Offside"]
            centers = [(t, "offside") for t in off]
            if confuser_per_half:
                far = [t for t, l in lab if l in CONFUSERS and all(abs(t - o) > 20 for o in off)]
                centers += [(float(t), "confuser") for t in rng.permutation(far)[:confuser_per_half]]
            if random_per_half:
                n_sec = len(np.load(g / f"{h}_ResNET_TF2_PCA512.npy", mmap_mode="r")) / FPS
                centers += [(float(t), "random") for t in rng.uniform(PRE_S, max(n_sec - POST_S, PRE_S + 1),
                                                                    random_per_half)]
            if candidates is not None:
                c = candidates[(candidates["game"] == str(g)) & (candidates["half"] == h)]
                centers += [(float(t), "candidate") for t in c["center_s"]]
            for t, kind in centers:
                rows.append({"clip_id": clip_id(str(g), h, t), "split": split, "game": str(g), "half": h,
                             "center_s": t, "kind": kind, "offsides": offsides_near(lab, t)})
    return pd.DataFrame(rows).drop_duplicates("clip_id").reset_index(drop=True)


def _extract_half(game: str, half: int, rows: list[dict], out_dir: Path) -> int:
    todo = [r for r in rows if not (out_dir / f"{r['clip_id']}.pkl").exists()]
    if todo:
        mid = (POST_S - PRE_S) / 2
        clips = read_windows(Path(game) / VIDEO_FILE.format(half=half), [r["center_s"] + mid for r in todo],
                             half_s=(PRE_S + POST_S) / 2, fps=CLIP_FPS)
        for r, frames in zip(todo, clips):
            (out_dir / f"{r['clip_id']}.pkl").write_bytes(pickle.dumps(frames))
    return len(todo)


def extract(plan: pd.DataFrame, workers: int = 12, out_dir: Path = CLIPS_DIR) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = list(plan.groupby(["game", "half"]))
    with ProcessPoolExecutor(workers) as pool:
        futs = [pool.submit(_extract_half, g, h, df.to_dict("records"), out_dir) for (g, h), df in groups]
        for i, f in enumerate(as_completed(futs), 1):
            f.result()
            if i % 50 == 0 or i == len(futs):
                log.info("extracted offside clips for %d/%d halves", i, len(futs))
    return plan


def load_clip(cid: str, out_dir: Path = CLIPS_DIR) -> list[bytes]:
    return pickle.loads((out_dir / f"{cid}.pkl").read_bytes())
