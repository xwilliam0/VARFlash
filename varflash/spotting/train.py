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
from .context_loss import ContextAwareLoss, bounds_for, time_shifts
from .data import (FOCUS_CLASSES, FPS, MARKET_MOVING, RARE_CLASSES, Half, WindowDataset, available_games,
                   class_map, load_halves)
from .infer import SpotterEnsemble, event_scores, frame_scores, load_spotter, peak_pick
from .metrics import (calibration_report, fit_isotonic, latency_report, match_predictions, pick_threshold,
                      spotting_report)
from .model import TemporalSpotter

log = logging.getLogger(__name__)

SELECT_TOLERANCES = (5, 10, 30, 60)


@dataclass
class TrainConfig:
    epochs: int = 30
    patience: int = 6  # epochs without a better selection score before stopping
    select_classes: list[str] | None = None  # classes whose valid mAP picks the epoch (default: focus classes)
    select_smooth: int = 1  # average the selection score over this many recent epochs
    batch_size: int = 64
    window: int = 120  # frames (60 s at 2 fps)
    # Frames at each end of a training window with no loss; inference then scores each frame from the
    # centre of a window, so every frame has >= edge frames of past and future context.
    edge: int = 0
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    label_radius: int = 2  # +-1 s around each event counts as positive
    label_sigma: float = 0.0  # > 0: Gaussian targets with this std (frames) instead of the +-radius box
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dim: int = 128
    dropout: float = 0.2
    max_pos_weight: float = 50.0
    classes: str = "all"  # key of data.CLASS_SETS
    loss: str = "bce"  # "bce", "focal", or "calf" (context-aware slicing)
    focal_gamma: float = 2.0
    oversample: int = 0  # extra windows per rare-class event (red cards, penalties)
    mixup_alpha: float = 0.0  # > 0: blend window pairs (features and targets) with lam ~ Beta(a, a)
    time_mask: int = 0  # > 0: zero two random spans of up to this many frames per training window
    ema: float = 0.0  # > 0: validate and save an exponential moving average of the weights with this decay
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


def augment(x: torch.Tensor, y: torch.Tensor, mixup_alpha: float = 0.0, time_mask: int = 0,
            n_masks: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Feature-space augmentation for (B, T, D) windows and (B, T, C) targets.

    Time masking hides short spans (replays and close-ups do the same in a broadcast) so
    the model has to use wider context; targets are kept. Mixup blends each window with
    another from the batch; multi-label sigmoid targets blend the same way.
    """
    B, T, _ = x.shape
    if time_mask > 0:
        t = torch.arange(T, device=x.device)
        for _ in range(n_masks):
            w = torch.randint(0, min(time_mask, T) + 1, (B,), device=x.device)
            s = (torch.rand(B, device=x.device) * (T - w + 1)).long()
            x = x.masked_fill(((t >= s[:, None]) & (t < (s + w)[:, None]))[..., None], 0.0)
    if mixup_alpha > 0 and B > 1:
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
        perm = torch.randperm(B, device=x.device)
        x = lam * x + (1 - lam) * x[perm]
        y = lam * y + (1 - lam) * y[perm]
    return x, y


class EMA:
    """Moving average of a model's weights and BatchNorm statistics (integer buffers are copied).

    The decay ramps up as ``(1 + t) / (10 + t)`` so the random init washes out within an epoch.
    """

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay, self.steps = decay, 0
        self.module = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))
        for e, m in zip(self.module.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.lerp_(m, 1 - d)
            else:
                e.copy_(m)


def _epoch(model, loader, loss_fn, device, opt=None, aug: dict | None = None, ema: EMA | None = None,
           edge: int = 0) -> float:
    model.train(opt is not None)
    total, n = 0.0, 0
    with torch.set_grad_enabled(opt is not None):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if opt is not None and aug:
                x, y = augment(x, y, **aug)
            core = slice(edge, x.shape[1] - edge)
            loss = loss_fn(model(x)[:, core], y[:, core])
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                if ema is not None:
                    ema.update(model)
            total += loss.item() * len(x)
            n += len(x)
    return total / max(n, 1)


def _video(h: Half) -> str:
    return f"{h.game}#{h.half}"


def predict_halves(model, halves: list[Half], classes: list[str], window: int, delay: int,
                   calibrators: dict | None = None, edge: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, list[np.ndarray]]:
    """Peak-picked predictions, ground truth and raw live scores for a list of halves."""
    model.eval()
    preds, gts, live = [], [], []
    for h in halves:
        raw = frame_scores(model, h.features, window, edge)
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


def fit_post_hoc(model, valid_halves: list[Half], classes: list[str], focus: list[str],
                 cfg: TrainConfig) -> tuple[dict, dict]:
    """Isotonic calibrators per class and (causal) live alert thresholds, fitted on validation only."""
    preds, gts, live = predict_halves(model, valid_halves, classes, cfg.window, cfg.delay, edge=cfg.edge)
    calibrators, thresholds = {}, {}
    for c in classes:
        p, g = preds[preds["label"] == c], gts[gts["label"] == c]
        if len(p) and len(g):
            calibrators[c] = fit_isotonic(p["confidence"].to_numpy(), match_predictions(p, g, cfg.calib_tolerance_s))
    if cfg.causal:
        for c in focus:
            thresholds[c] = pick_threshold(_live_inputs(valid_halves, live, classes.index(c)), cfg.live_precision)
    return calibrators, thresholds


def load_ensemble(paths: list[Path | str], device: str = "cpu") -> tuple[SpotterEnsemble, dict]:
    """Average several spotters; calibrators and live thresholds are refitted on validation for the ensemble."""
    members = [load_spotter(p, device) for p in paths]
    ckpt = members[0][1]
    for p, (_, c) in zip(paths, members):
        if "members" in c:
            raise ValueError(f"{p} is already an ensemble; pass its member checkpoints instead")
        for key in ("classes", "delay"):
            if c[key] != ckpt[key]:
                raise ValueError(f"{p}: {key} {c[key]} differs from {paths[0]}: {ckpt[key]}")
        for key in ("window", "edge"):
            if c["config"].get(key) != ckpt["config"].get(key):
                raise ValueError(f"{p}: {key} differs from {paths[0]}")
    model = SpotterEnsemble([m for m, _ in members]).eval()
    cfg = TrainConfig(**{k: v for k, v in ckpt["config"].items() if k in TrainConfig.__dataclass_fields__})
    valid_halves = load_halves(available_games("valid"), radius=cfg.label_radius, classes=cfg.classes,
                               delay=cfg.delay)
    focus = ckpt.get("focus") or ckpt["classes"]
    calibrators, thresholds = fit_post_hoc(model, valid_halves, ckpt["classes"], focus, cfg)
    preds, gts, _ = predict_halves(model, valid_halves, ckpt["classes"], cfg.window, cfg.delay, edge=cfg.edge)
    valid_map = _mean_ap(spotting_report(preds, gts, ckpt["classes"], SELECT_TOLERANCES), focus)
    meta = {k: v for k, v in ckpt.items() if k not in ("state_dict", "model_args", "history")}
    return model, {**meta, "members": [{"path": str(p), "model_args": c["model_args"]} for p, (_, c) in
                                       zip(paths, members)],
                   "calibrators": calibrators, "live_thresholds": thresholds, "valid_map": valid_map, "epoch": None}


def save_ensemble(paths: list[Path | str], out: Path, device: str = "cpu") -> dict:
    """One self-contained checkpoint (member weights embedded) that ``load_spotter`` opens as an ensemble."""
    model, ckpt = load_ensemble(paths, device)
    for m, member in zip(model.models, ckpt["members"]):
        member["state_dict"] = {k: v.cpu() for k, v in m.state_dict().items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    log.info("ensemble of %d (valid mAP %.4f) -> %s", len(paths), ckpt["valid_map"], out)
    return ckpt


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

    load = dict(radius=cfg.label_radius, classes=cfg.classes, delay=cfg.delay, in_memory=True, sigma=cfg.label_sigma)
    train_halves = load_halves(train_games, **load)
    valid_halves = load_halves(valid_games, **load)
    context_bounds = bounds_for(classes) if cfg.loss == "calf" else None
    if context_bounds is not None:
        # Integer time-shifts replace the ±radius bits. The datasets read them
        # off each Half, so they have to be filled before the first batch.
        for halves in (train_halves, valid_halves):
            for h in halves:
                h.shifts = time_shifts(len(h.features), h.events, context_bounds, cfg.delay)
    rare = [i for i, c in enumerate(classes) if c in RARE_CLASSES]
    train_ds = WindowDataset(train_halves, cfg.window, train=True, oversample=cfg.oversample, oversample_classes=rare,
                             pad=cfg.edge)
    loader = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
                  persistent_workers=cfg.num_workers > 0)
    train_dl = DataLoader(train_ds, shuffle=True, **loader)
    valid_dl = DataLoader(WindowDataset(valid_halves, cfg.window, stride=cfg.window - 2 * cfg.edge, train=False,
                                        pad=cfg.edge), **loader)

    model_args = {"in_dim": 512, "dim": cfg.dim, "n_classes": len(classes), "dropout": cfg.dropout,
                  "causal": cfg.causal, "dilations": list(cfg.dilations)}
    model = TemporalSpotter(**model_args).to(device)
    if cfg.loss == "calf":
        # Mixup would average integer time-shifts into nonsense, so it stays off.
        loss_fn = ContextAwareLoss(context_bounds).to(device)
        if cfg.mixup_alpha:
            log.info("context-aware loss does not mix targets; ignoring mixup_alpha")
    elif cfg.loss == "focal":
        loss_fn = FocalBCE(_pos_weight(train_halves, cfg.max_pos_weight), cfg.focal_gamma).to(device)
    elif cfg.loss == "bce":
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=_pos_weight(train_halves, cfg.max_pos_weight)).to(device)
    else:
        raise ValueError(f"unknown loss {cfg.loss!r}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)

    aug = {"mixup_alpha": 0.0 if cfg.loss == "calf" else cfg.mixup_alpha, "time_mask": cfg.time_mask}
    ema = EMA(model, cfg.ema) if cfg.ema > 0 else None
    scored = ema.module if ema else model  # what gets validated and saved
    select = [c for c in cfg.select_classes if c in classes] if cfg.select_classes else focus
    if cfg.select_classes and len(select) != len(cfg.select_classes):
        raise ValueError(f"select_classes {cfg.select_classes} not all in {classes}")
    best, best_state, best_epoch, best_vmap, history = -1.0, None, 0, float("nan"), []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr = _epoch(model, train_dl, loss_fn, device, opt, aug, ema, cfg.edge)
        va = _epoch(scored, valid_dl, loss_fn, device, edge=cfg.edge)
        sched.step()
        preds, gts, _ = predict_halves(scored, valid_halves, classes, cfg.window, cfg.delay, edge=cfg.edge)
        report = spotting_report(preds, gts, classes, SELECT_TOLERANCES)
        vmap = _mean_ap(report, focus)
        per_class = {r["class"]: round(float(np.nanmean([r[f"AP@{t}s"] for t in SELECT_TOLERANCES])), 3)
                     for _, r in report[report["class"].isin(focus)].iterrows()} if len(focus) <= 4 else {}
        sel = vmap if select == focus else _mean_ap(report, select)
        history.append({"epoch": epoch, "train_loss": tr, "valid_loss": va, "valid_map": vmap, "select": sel})
        # Smoothed over the last select_smooth epochs: one lucky epoch on a rare class can't win alone.
        score = float(np.mean([h["select"] for h in history[-cfg.select_smooth:]]))
        log.info("epoch %2d  train %.4f  valid %.4f  valid mAP %.4f %s select %.4f (%.0fs)", epoch, tr, va, vmap,
                 per_class or "", score, time.time() - t0)
        if score > best:
            best, best_epoch, best_vmap, best_state = score, epoch, vmap, copy.deepcopy(scored.state_dict())
        elif epoch - best_epoch >= cfg.patience:
            log.info("early stop: no improvement for %d epochs", cfg.patience)
            break
    model.load_state_dict(best_state)
    calibrators, thresholds = fit_post_hoc(model, valid_halves, classes, focus, cfg)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_args": model_args, "config": asdict(cfg),
                "classes": classes, "focus": focus, "delay": cfg.delay, "valid_map": best_vmap, "epoch": best_epoch,
                "select_score": best, "history": history, "calibrators": calibrators,
                "live_thresholds": thresholds}, out)
    log.info("best valid mAP %.4f (epoch %d, select %.4f) -> %s", best_vmap, best_epoch, best, out)
    return out


def evaluate(model: TemporalSpotter, split: str = "test", window: int = 120, max_games: int | None = None,
             ckpt: dict | None = None, edge: int | None = None) -> dict[str, pd.DataFrame]:
    """AP per class, calibration before/after, and (causal models) live alert latency."""
    ckpt = ckpt or {}
    cfg = ckpt.get("config", {})
    classes = ckpt.get("classes") or list(class_map("all"))
    delay = ckpt.get("delay", 0)
    window = cfg.get("window", window)
    halves = load_halves(available_games(split)[:max_games], radius=0, classes=cfg.get("classes", "all"))
    edge = cfg.get("edge", 0) if edge is None else edge
    preds, gts, live = predict_halves(model, halves, classes, window, delay, ckpt.get("calibrators"), edge)

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
