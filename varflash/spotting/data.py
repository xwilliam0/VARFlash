"""SoccerNet-v2 action-spotting data: download, frame labels and training windows.

SoccerNet publishes, for 500 full broadcast matches, per-frame visual features
(ResNet-152 at 2 fps, PCA-reduced to 512-d) and timestamped event labels.
Features and labels are freely downloadable; raw videos need SoccerNet's NDA.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..paths import SOCCERNET_DATA

log = logging.getLogger(__name__)

FPS = 2
FEATURE_FILE = "{half}_ResNET_TF2_PCA512.npy"
LABEL_FILE = "Labels-v2.json"
CLASSES = [
    "Penalty",
    "Kick-off",
    "Goal",
    "Substitution",
    "Offside",
    "Shots on target",
    "Shots off target",
    "Clearance",
    "Ball out of play",
    "Throw-in",
    "Foul",
    "Indirect free-kick",
    "Direct free-kick",
    "Corner",
    "Yellow card",
    "Red card",
    "Yellow->red card",
]
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
# Events that move a match-winner market; reported separately in evaluation.
MARKET_MOVING = ["Goal", "Penalty", "Red card", "Yellow->red card", "Offside"]


def list_games(split: str) -> list[str]:
    from SoccerNet.utils import getListGames

    return getListGames(split)


def download(
    splits=("train", "valid", "test"),
    max_games: int | None = None,
    workers: int = 8,
    root: Path = SOCCERNET_DATA,
):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from SoccerNet.Downloader import SoccerNetDownloader

    logging.getLogger("httpx").setLevel(logging.WARNING)
    files = [LABEL_FILE, FEATURE_FILE.format(half=1), FEATURE_FILE.format(half=2)]
    todo = [
        g
        for split in splits
        for g in list_games(split)[:max_games]
        if not all((root / g / f).exists() for f in files)
    ]
    log.info("%d games to download", len(todo))

    def fetch(game, attempts: int = 6):
        # Unauthenticated Hugging Face requests get rate-limited; back off and retry.
        for i in range(attempts):
            try:
                SoccerNetDownloader(LocalDirectory=str(root)).downloadGame(game=game, files=files)
                if all((root / game / f).exists() for f in files):
                    return game
            except Exception:
                if i == attempts - 1:
                    raise
            time.sleep(min(30 * 2**i, 300))
        raise RuntimeError(f"incomplete after {attempts} attempts: {game}")

    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(fetch, g) for g in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                log.info("[%d/%d] %s", i, len(todo), fut.result())
            except Exception as exc:
                log.error("download failed: %s", exc)


def available_games(split: str, root: Path = SOCCERNET_DATA) -> list[Path]:
    out = []
    for g in list_games(split):
        d = root / g
        if (d / LABEL_FILE).exists() and (d / FEATURE_FILE.format(half=2)).exists():
            out.append(d)
    return out


def half_events(game_dir: Path) -> dict[int, list[tuple[int, int]]]:
    """{half: [(frame, class_idx), ...]} from Labels-v2.json."""
    labels = json.loads((game_dir / LABEL_FILE).read_text())
    out: dict[int, list[tuple[int, int]]] = {1: [], 2: []}
    for a in labels["annotations"]:
        if a["label"] not in CLASS_INDEX:
            continue
        half = int(a["gameTime"].split(" - ")[0])
        frame = int(int(a["position"]) / 1000 * FPS)
        out[half].append((frame, CLASS_INDEX[a["label"]]))
    return out


def frame_targets(n_frames: int, events: list[tuple[int, int]], radius: int) -> np.ndarray:
    """Binary (n_frames, C) targets, 1 within ``radius`` frames of each event."""
    y = np.zeros((n_frames, len(CLASSES)), np.float32)
    for f, c in events:
        y[max(f - radius, 0) : min(f + radius + 1, n_frames), c] = 1.0
    return y


@dataclass
class Half:
    game: str
    half: int
    features: np.ndarray  # memory-mapped (N, 512)
    targets: np.ndarray  # (N, C)
    events: list[tuple[int, int]]


def load_halves(game_dirs: list[Path], radius: int = 2) -> list[Half]:
    halves = []
    for g in game_dirs:
        ev = half_events(g)
        for h in (1, 2):
            feats = np.load(g / FEATURE_FILE.format(half=h), mmap_mode="r")
            halves.append(Half(str(g), h, feats, frame_targets(len(feats), ev[h], radius), ev[h]))
    return halves


class WindowDataset(Dataset):
    """Fixed-length windows over match halves; random offsets when ``train``."""

    def __init__(self, halves: list[Half], window: int = 120, stride: int | None = None, train: bool = True):
        self.halves = halves
        self.window = window
        self.train = train
        stride = stride or window // 2
        self.index = [
            (i, s) for i, h in enumerate(halves) for s in range(0, max(len(h.features) - window, 0) + 1, stride)
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, s = self.index[k]
        h = self.halves[i]
        if self.train:
            s = int(np.clip(s + np.random.randint(-self.window // 2, self.window // 2 + 1), 0, len(h.features) - self.window))
        x = np.array(h.features[s : s + self.window], np.float32)
        y = h.targets[s : s + self.window]
        return torch.from_numpy(x), torch.from_numpy(y)
