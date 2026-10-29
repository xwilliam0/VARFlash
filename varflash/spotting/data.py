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
MARKET_MOVING = ["Goal", "Penalty", "Red card", "Yellow->red card", "Offside", "Sending off"]

# Model output class -> SoccerNet labels it covers. "cards" merges both ways of being sent
# off (only ~58 in train) and keeps Foul as an auxiliary task: cards follow fouls.
CLASS_SETS: dict[str, dict[str, list[str]]] = {
    "all": {c: [c] for c in CLASSES},
    "cards": {"Yellow card": ["Yellow card"], "Sending off": ["Red card", "Yellow->red card"], "Foul": ["Foul"]},
    # Every SoccerNet class as auxiliary supervision, both reds merged, selected on cards.
    "cards_multitask": {**{c: [c] for c in CLASSES if c not in ("Red card", "Yellow->red card")},
                        "Sending off": ["Red card", "Yellow->red card"]},
}
# Classes used for model selection / headline metrics of each set.
FOCUS_CLASSES = {"all": CLASSES, "cards": ["Yellow card", "Sending off"],
                 "cards_multitask": ["Yellow card", "Sending off"]}
RARE_CLASSES = ["Red card", "Yellow->red card", "Sending off", "Penalty"]


def class_map(classes: str | dict[str, list[str]] = "all") -> dict[str, list[str]]:
    return CLASS_SETS[classes] if isinstance(classes, str) else classes


def list_games(split: str) -> list[str]:
    from SoccerNet.utils import getListGames

    return getListGames(split)


def download(
    splits=("train", "valid", "test"),
    max_games: int | None = None,
    workers: int = 8,
    root: Path = SOCCERNET_DATA,
    files: list[str] | None = None,
):
    """Per-game SoccerNet files. Default: labels + ResNet PCA-512 features.

    Other useful ``files``: ``1_ResNET_TF2.npy`` (raw 2048-d) or ``1_224p.mkv``
    (video, needs the NDA-granted Hugging Face login).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from SoccerNet.Downloader import SoccerNetDownloader

    logging.getLogger("httpx").setLevel(logging.WARNING)
    files = files or [LABEL_FILE, FEATURE_FILE.format(half=1), FEATURE_FILE.format(half=2)]
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


def half_events(game_dir: Path, classes: str | dict[str, list[str]] = "all") -> dict[int, list[tuple[int, int]]]:
    """{half: [(frame, class_idx), ...]} from Labels-v2.json, indexed into ``classes``."""
    cmap = class_map(classes)
    index = {src: i for i, srcs in enumerate(cmap.values()) for src in srcs}
    labels = json.loads((game_dir / LABEL_FILE).read_text())
    out: dict[int, list[tuple[int, int]]] = {1: [], 2: []}
    for a in labels["annotations"]:
        if a["label"] not in index:
            continue
        half = int(a["gameTime"].split(" - ")[0])
        frame = int(int(a["position"]) / 1000 * FPS)
        out[half].append((frame, index[a["label"]]))
    return out


def frame_targets(n_frames: int, events: list[tuple[int, int]], radius: int, n_classes: int = len(CLASSES),
                  delay: int = 0) -> np.ndarray:
    """Binary (n_frames, C) targets, 1 within ``radius`` frames of each event.

    ``delay`` > 0 moves each target ``delay`` frames later: a causal model then has
    ``delay`` frames of look-ahead before it must report the event.
    """
    y = np.zeros((n_frames, n_classes), np.float32)
    for f, c in events:
        f += delay
        y[max(f - radius, 0) : max(min(f + radius + 1, n_frames), 0), c] = 1.0
    return y


@dataclass
class Half:
    game: str
    half: int
    features: np.ndarray  # (N, 512), memory-mapped or in RAM
    targets: np.ndarray  # (N, C)
    events: list[tuple[int, int]]  # (frame, class_idx), at the true event time (no delay)


def load_halves(game_dirs: list[Path], radius: int = 2, classes: str | dict[str, list[str]] = "all",
                delay: int = 0, in_memory: bool = False) -> list[Half]:
    n_classes = len(class_map(classes))
    halves = []
    for g in game_dirs:
        ev = half_events(g, classes)
        for h in (1, 2):
            feats = np.load(g / FEATURE_FILE.format(half=h), mmap_mode=None if in_memory else "r")
            halves.append(Half(str(g), h, feats, frame_targets(len(feats), ev[h], radius, n_classes, delay), ev[h]))
    return halves


class WindowDataset(Dataset):
    """Fixed-length windows over match halves; random offsets when ``train``.

    ``oversample`` adds that many extra windows around every event of a class in
    ``oversample_classes`` (rare cards), so each epoch sees them far more often.
    """

    def __init__(self, halves: list[Half], window: int = 120, stride: int | None = None, train: bool = True,
                 oversample: int = 0, oversample_classes: list[int] = ()):
        self.halves = halves
        self.window = window
        self.train = train
        stride = stride or window // 2
        self.index = [
            (i, s) for i, h in enumerate(halves) for s in range(0, max(len(h.features) - window, 0) + 1, stride)
        ]
        if train and oversample:
            for i, h in enumerate(halves):
                for f, c in h.events:
                    if c in oversample_classes:
                        self.index += [(i, max(f - window // 2, 0))] * oversample

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, s = self.index[k]
        h = self.halves[i]
        if self.train:
            j = self.window * 2 // 5  # keeps an oversampled event inside its window
            s = int(np.clip(s + np.random.randint(-j, j + 1), 0, max(len(h.features) - self.window, 0)))
        x = np.array(h.features[s : s + self.window], np.float32)
        y = h.targets[s : s + self.window]
        return torch.from_numpy(x), torch.from_numpy(y)
