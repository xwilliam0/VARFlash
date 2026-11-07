"""Raw-frame clips around shots in SoccerNet broadcasts, for the visual goal detector.

A shot "episode" is one or more shot / penalty / goal annotations within a few seconds of
each other; it is a goal episode if a Goal is annotated during it. Saved and missed shots
(over the bar, side netting, off the post) are the hard negatives; random moments away from
shots are easy negatives. Clips run from 4 s before to 6 s after the first shot at 12.5 fps,
fast enough to see the ball cross the line and the net move, and are stored as JPEG bytes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from ..cards.clips import VIDEO_FILE, read_windows
from ..paths import SOCCERNET_DATA
from ..spotting.data import FPS, LABEL_FILE, available_games

log = logging.getLogger(__name__)

CLIP_FPS = 12.5
PRE_S, POST_S = 4.0, 6.0  # stored window around the first shot of an episode
CLIP_FRAMES = int((PRE_S + POST_S) * CLIP_FPS)  # 125
CLIPS_DIR = SOCCERNET_DATA / "goal_clips"
SHOT_LABELS = ("Shots on target", "Shots off target", "Penalty")
MERGE_S = 4.0  # shots this close belong to one episode (rebounds, follow-ups)
GOAL_AFTER_S = 5.0  # a Goal up to this long after the last shot belongs to the episode
NO_SHOT_GOAL_LEAD_S = 0.6  # goals without an annotated shot: anchor this much earlier


def half_labels(game_dir: Path) -> dict[int, list[tuple[float, str]]]:
    """{half: [(time_s, label)]} with SoccerNet's millisecond positions."""
    labels = json.loads((game_dir / LABEL_FILE).read_text())
    out: dict[int, list[tuple[float, str]]] = {1: [], 2: []}
    for a in labels["annotations"]:
        out[int(a["gameTime"].split(" - ")[0])].append((int(a["position"]) / 1000, a["label"]))
    return {h: sorted(v) for h, v in out.items()}


def shot_episodes(events: list[tuple[float, str]]) -> list[dict]:
    """Group shots into episodes; ``goal_s`` is the first Goal annotated during one (else NaN)."""
    goals = [t for t, l in events if l == "Goal"]
    shots = [(t, l) for t, l in events if l in SHOT_LABELS]
    anchors = shots + [(g - NO_SHOT_GOAL_LEAD_S, "Goal") for g in goals
                       if not any(g - 3.0 <= t <= g for t, _ in shots)]
    episodes: list[dict] = []
    for t, l in sorted(anchors):
        if episodes and t - episodes[-1]["last_s"] <= MERGE_S:
            episodes[-1]["last_s"] = t
            episodes[-1]["labels"].append(l)
        else:
            episodes.append({"anchor_s": t, "last_s": t, "labels": [l]})
    for e in episodes:
        g = [x for x in goals if e["anchor_s"] - 1.0 <= x <= e["last_s"] + GOAL_AFTER_S]
        e["goal_s"] = g[0] if g else float("nan")
        e["kind"] = ("goal" if g else "penalty_miss" if "Penalty" in e["labels"]
                     else "shot_on" if "Shots on target" in e["labels"] else "shot_off")
        e["labels"] = "|".join(e["labels"])
    return episodes


def clip_id(game: str, half: int, anchor_s: float) -> str:
    g = hashlib.md5(Path(game).relative_to(SOCCERNET_DATA).as_posix().encode()).hexdigest()[:12]
    return f"{g}_{half}_{int(round(anchor_s * 1000)):08d}"


def plan_clips(split: str, random_per_half: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in available_games(split):
        if not all((g / VIDEO_FILE.format(half=h)).exists() for h in (1, 2)):
            continue
        labels = half_labels(g)
        for h in (1, 2):
            eps = shot_episodes(labels[h])
            n_sec = len(np.load(g / f"{h}_ResNET_TF2_PCA512.npy", mmap_mode="r")) / FPS
            goals = [t for t, l in labels[h] if l == "Goal"]
            shots = [e["anchor_s"] for e in eps]
            # Random negatives away from shots and from the replays that follow a goal.
            cand = rng.uniform(PRE_S + 1, max(n_sec - POST_S - 1, PRE_S + 2), 20 * random_per_half)
            ok = [t for t in cand if all(abs(t - s) > 15 for s in shots) and all(not (0 <= t - x <= 120) for x in goals)]
            eps += [{"anchor_s": float(t), "last_s": float(t), "labels": "", "goal_s": float("nan"), "kind": "random"}
                    for t in ok[:random_per_half]]
            for e in eps:
                rows.append({"clip_id": clip_id(str(g), h, e["anchor_s"]), "split": split, "game": str(g), "half": h,
                             "anchor_s": e["anchor_s"], "start_s": e["anchor_s"] - PRE_S, "goal_s": e["goal_s"],
                             "kind": e["kind"], "labels": e["labels"]})
    return pd.DataFrame(rows).drop_duplicates("clip_id").reset_index(drop=True)


def _extract_half(game: str, half: int, rows: list[dict], out_dir: Path) -> int:
    todo = [r for r in rows if not (out_dir / f"{r['clip_id']}.pkl").exists()]
    if todo:
        half_s = (PRE_S + POST_S) / 2
        clips = read_windows(Path(game) / VIDEO_FILE.format(half=half), [r["start_s"] + half_s for r in todo],
                             half_s=half_s, fps=CLIP_FPS)
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
                log.info("extracted goal clips for %d/%d halves", i, len(futs))
    return plan


def load_clip(cid: str, out_dir: Path = CLIPS_DIR) -> list[bytes]:
    return pickle.loads((out_dir / f"{cid}.pkl").read_bytes())
