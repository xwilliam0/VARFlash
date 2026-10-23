"""Train / evaluate the temporal spotter on SoccerNet features (CPU-friendly)."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..paths import MODELS
from .data import CLASSES, FPS, MARKET_MOVING, WindowDataset, available_games, load_halves
from .infer import frame_scores, peak_pick
from .metrics import spotting_report
from .model import TemporalSpotter

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 32
    window: int = 120  # frames (60 s at 2 fps)
    label_radius: int = 2  # +-1 s around each event counts as positive
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dim: int = 128
    dropout: float = 0.2
    max_pos_weight: float = 50.0
    seed: int = 0
    max_train_games: int | None = None
    num_threads: int | None = None


def _pos_weight(halves, cap: float) -> torch.Tensor:
    pos = sum(h.targets.sum(axis=0) for h in halves)
    total = sum(len(h.targets) for h in halves)
    w = (total - pos) / np.maximum(pos, 1)
    return torch.tensor(np.minimum(w, cap), dtype=torch.float32)


def _epoch(model, loader, loss_fn, opt=None) -> float:
    model.train(opt is not None)
    total, n = 0.0, 0
    with torch.set_grad_enabled(opt is not None):
        for x, y in loader:
            loss = loss_fn(model(x), y)
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            total += loss.item() * len(x)
            n += len(x)
    return total / max(n, 1)


def train(cfg: TrainConfig, out: Path = MODELS / "spotter.pt") -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.num_threads:
        torch.set_num_threads(cfg.num_threads)

    train_games = available_games("train")[: cfg.max_train_games]
    valid_games = available_games("valid")
    if not train_games or not valid_games:
        raise RuntimeError("no SoccerNet data found; run `varflash spotting download` first")
    log.info("train games: %d, valid games: %d", len(train_games), len(valid_games))

    train_halves = load_halves(train_games, cfg.label_radius)
    valid_halves = load_halves(valid_games, cfg.label_radius)
    train_dl = DataLoader(WindowDataset(train_halves, cfg.window, train=True), batch_size=cfg.batch_size, shuffle=True)
    valid_dl = DataLoader(WindowDataset(valid_halves, cfg.window, stride=cfg.window, train=False), batch_size=cfg.batch_size)

    model_args = {"in_dim": 512, "dim": cfg.dim, "n_classes": len(CLASSES), "dropout": cfg.dropout}
    model = TemporalSpotter(**model_args)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=_pos_weight(train_halves, cfg.max_pos_weight))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)

    out.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr = _epoch(model, train_dl, loss_fn, opt)
        va = _epoch(model, valid_dl, loss_fn)
        sched.step()
        log.info("epoch %2d  train %.4f  valid %.4f  (%.0fs)", epoch, tr, va, time.time() - t0)
        if va < best:
            best = va
            torch.save(
                {"state_dict": model.state_dict(), "model_args": model_args, "config": asdict(cfg),
                 "classes": CLASSES, "valid_loss": va, "epoch": epoch},
                out,
            )
    log.info("best valid loss %.4f -> %s", best, out)
    return out


def evaluate(model: TemporalSpotter, split: str = "test", window: int = 120, max_games: int | None = None) -> pd.DataFrame:
    games = available_games(split)[:max_games]
    halves = load_halves(games, radius=0)
    preds, gts = [], []
    for h in halves:
        video = f"{h.game}#{h.half}"
        p = peak_pick(frame_scores(model, h.features, window))
        p["video"] = video
        preds.append(p)
        gts.append(pd.DataFrame([{"video": video, "time_s": f / FPS, "label": CLASSES[c]} for f, c in h.events]))
    preds_df, gts_df = pd.concat(preds, ignore_index=True), pd.concat(gts, ignore_index=True)
    report = spotting_report(preds_df, gts_df, CLASSES)
    report["market_moving"] = report["class"].isin(MARKET_MOVING)
    return report
