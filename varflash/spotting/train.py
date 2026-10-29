"""Train / evaluate the temporal spotter on SoccerNet features (GPU if available, CPU-friendly)."""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..paths import MODELS
from .data import (FOCUS_CLASSES, FPS, MARKET_MOVING, RARE_CLASSES, Half, WindowDataset, available_games,
                   class_map, load_halves)
from .infer import event_scores, frame_scores, peak_pick
from .metrics import (calibration_report, fit_isotonic, latency_report, match_predictions, pick_threshold,
                      spotting_report)
from .model import TemporalSpotter

log = logging.getLogger(__name__)

SELECT_TOLERANCES = (5, 10, 30, 60)


@dataclass
class TrainConfig:
    epochs: int = 30
    patience: int = 6  # epochs without a better validation mAP before stopping
    batch_size: int = 64
    window: int = 120  # frames (60 s at 2 fps)
    label_radius: int = 2  # +-1 s around each event counts as positive
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dim: int = 128
    dropout: float = 0.2
    max_pos_weight: float = 50.0
    classes: str = "all"  # key of data.CLASS_SETS
    loss: str = "bce"  # "bce" or "focal"
    focal_gamma: float = 2.0
    oversample: int = 0  # extra windows per rare-class event (red cards, penalties)
    causal: bool = False
    delay_s: float = 0.0  # causal look-ahead budget: targets are shifted this much later
    calib_tolerance_s: float = 10.0  # a peak within this of a real event counts as "true" for calibration
    live_precision: float = 0.5  # alert threshold = lowest with this live precision on valid
    seed: int = 0
    max_train_games: int | None = None
    num_threads: int | None = None
    num_workers: int = 4
    device: str | None = None

    @property
    def delay(self) -> int:
        return int(round(self.delay_s * FPS))


class FocalBCE(torch.nn.Module):
    """Class-weighted BCE scaled by (1 - p_t)^gamma: easy frames (most of the match) count less."""

    def __init__(self, pos_weight: torch.Tensor, gamma: float):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.gamma = gamma

    def forward(self, logits, y):
        bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight, reduction="none")
        p = torch.sigmoid(logits)
        pt = y * p + (1 - y) * (1 - p)
        return ((1 - pt) ** self.gamma * bce).mean()


def _pos_weight(halves, cap: float) -> torch.Tensor:
    pos = sum(h.targets.sum(axis=0) for h in halves)
    total = sum(len(h.targets) for h in halves)
    w = (total - pos) / np.maximum(pos, 1)
    return torch.tensor(np.minimum(w, cap), dtype=torch.float32)


def _epoch(model, loader, loss_fn, device, opt=None) -> float:
    model.train(opt is not None)
    total, n = 0.0, 0
    with torch.set_grad_enabled(opt is not None):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = loss_fn(model(x), y)
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            total += loss.item() * len(x)
            n += len(x)
    return total / max(n, 1)


def _video(h: Half) -> str:
    return f"{h.game}#{h.half}"


def predict_halves(model, halves: list[Half], classes: list[str], window: int, delay: int,
                   calibrators: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[np.ndarray]]:
    """Peak-picked predictions, ground truth and raw live scores for a list of halves."""
    model.eval()
    preds, gts, live = [], [], []
    for h in halves:
        raw = frame_scores(model, h.features, window)
        live.append(raw)
        p = peak_pick(event_scores(raw, delay), classes, calibrators=calibrators)
        p["video"] = _video(h)
        preds.append(p)
        gts.append(pd.DataFrame([{"video": _video(h), "time_s": f / FPS, "label": classes[c]} for f, c in h.events],
                                columns=["video", "time_s", "label"]))
    return pd.concat(preds, ignore_index=True), pd.concat(gts, ignore_index=True), live


def _mean_ap(report: pd.DataFrame, focus: list[str]) -> float:
    cols = [f"AP@{t}s" for t in SELECT_TOLERANCES]
    return float(report[report["class"].isin(focus)][cols].mean().mean())


def _live_inputs(halves: list[Half], live: list[np.ndarray], c: int) -> list:
    return [(_video(h), s[:, c], np.array([f / FPS for f, k in h.events if k == c])) for h, s in zip(halves, live)]


def train(cfg: TrainConfig, out: Path = MODELS / "spotter.pt") -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.num_threads:
        torch.set_num_threads(cfg.num_threads)
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if cfg.delay and not cfg.causal:
        raise ValueError("delay_s only makes sense with causal=True")
    classes = list(class_map(cfg.classes))
    focus = [c for c in FOCUS_CLASSES.get(cfg.classes, classes) if c in classes]

    train_games = available_games("train")[: cfg.max_train_games]
    valid_games = available_games("valid")
    if not train_games or not valid_games:
        raise RuntimeError("no SoccerNet data found; run `varflash spotting download` first")
    log.info("train games: %d, valid games: %d, classes: %s, device: %s", len(train_games), len(valid_games),
             classes, device)

    load = dict(radius=cfg.label_radius, classes=cfg.classes, delay=cfg.delay, in_memory=True)
    train_halves = load_halves(train_games, **load)
    valid_halves = load_halves(valid_games, **load)
    rare = [i for i, c in enumerate(classes) if c in RARE_CLASSES]
    train_ds = WindowDataset(train_halves, cfg.window, train=True, oversample=cfg.oversample, oversample_classes=rare)
    loader = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
                  persistent_workers=cfg.num_workers > 0)
    train_dl = DataLoader(train_ds, shuffle=True, **loader)
    valid_dl = DataLoader(WindowDataset(valid_halves, cfg.window, stride=cfg.window, train=False), **loader)

    model_args = {"in_dim": 512, "dim": cfg.dim, "n_classes": len(classes), "dropout": cfg.dropout,
                  "causal": cfg.causal}
    model = TemporalSpotter(**model_args).to(device)
    pw = _pos_weight(train_halves, cfg.max_pos_weight)
    loss_fn = (FocalBCE(pw, cfg.focal_gamma) if cfg.loss == "focal"
               else torch.nn.BCEWithLogitsLoss(pos_weight=pw)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)

    best, best_state, best_epoch, history = -1.0, None, 0, []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr = _epoch(model, train_dl, loss_fn, device, opt)
        va = _epoch(model, valid_dl, loss_fn, device)
        sched.step()
        preds, gts, _ = predict_halves(model, valid_halves, classes, cfg.window, cfg.delay)
        report = spotting_report(preds, gts, classes, SELECT_TOLERANCES)
        vmap = _mean_ap(report, focus)
        per_class = {r["class"]: round(float(np.nanmean([r[f"AP@{t}s"] for t in SELECT_TOLERANCES])), 3)
                     for _, r in report[report["class"].isin(focus)].iterrows()} if len(focus) <= 4 else {}
        history.append({"epoch": epoch, "train_loss": tr, "valid_loss": va, "valid_map": vmap})
        log.info("epoch %2d  train %.4f  valid %.4f  valid mAP %.4f %s (%.0fs)", epoch, tr, va, vmap,
                 per_class or "", time.time() - t0)
        if vmap > best:
            best, best_epoch, best_state = vmap, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= cfg.patience:
            log.info("early stop: no improvement for %d epochs", cfg.patience)
            break
    model.load_state_dict(best_state)

    # Post-hoc calibration and live alert thresholds, both fitted on validation only.
    preds, gts, live = predict_halves(model, valid_halves, classes, cfg.window, cfg.delay)
    calibrators, thresholds = {}, {}
    for c in classes:
        p, g = preds[preds["label"] == c], gts[gts["label"] == c]
        if len(p) and len(g):
            calibrators[c] = fit_isotonic(p["confidence"].to_numpy(), match_predictions(p, g, cfg.calib_tolerance_s))
    if cfg.causal:
        for c in focus:
            thresholds[c] = pick_threshold(_live_inputs(valid_halves, live, classes.index(c)), cfg.live_precision)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_args": model_args, "config": asdict(cfg),
                "classes": classes, "focus": focus, "delay": cfg.delay, "valid_map": best, "epoch": best_epoch,
                "history": history, "calibrators": calibrators, "live_thresholds": thresholds}, out)
    log.info("best valid mAP %.4f (epoch %d) -> %s", best, best_epoch, out)
    return out


def evaluate(model: TemporalSpotter, split: str = "test", window: int = 120, max_games: int | None = None,
             ckpt: dict | None = None) -> dict[str, pd.DataFrame]:
    """AP per class, calibration before/after, and (causal models) live alert latency."""
    ckpt = ckpt or {}
    cfg = ckpt.get("config", {})
    classes = ckpt.get("classes") or list(class_map("all"))
    delay = ckpt.get("delay", 0)
    window = cfg.get("window", window)
    halves = load_halves(available_games(split)[:max_games], radius=0, classes=cfg.get("classes", "all"))
    preds, gts, live = predict_halves(model, halves, classes, window, delay, ckpt.get("calibrators"))

    report = spotting_report(preds, gts, classes)
    report["market_moving"] = report["class"].isin(MARKET_MOVING)
    out = {"ap": report}

    tol = cfg.get("calib_tolerance_s", 10.0)
    rows = []
    for c in ckpt.get("focus", classes):
        p, g = preds[preds["label"] == c], gts[gts["label"] == c]
        if p.empty or g.empty:
            continue
        hit = match_predictions(p, g, tol)
        rows.append({"class": c, "which": "raw", **calibration_report(p["raw_score"].to_numpy(), hit)})
        rows.append({"class": c, "which": "calibrated", **calibration_report(p["confidence"].to_numpy(), hit)})
    out["calibration"] = pd.DataFrame(rows)

    if cfg.get("causal"):
        rows = []
        for c, th in ckpt.get("live_thresholds", {}).items():
            r = latency_report(_live_inputs(halves, live, classes.index(c)), th)
            rows.append({"class": c, "delay_budget_s": delay / FPS, **r})
        out["latency"] = pd.DataFrame(rows)
    return out
