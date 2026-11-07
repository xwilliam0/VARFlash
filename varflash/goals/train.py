"""Train / evaluate the causal visual goal detector on pre-extracted shot clips.

The model emits, at every frame, P(a goal has been scored by now) using only the last
``frames`` frames (a forward-only GRU over a fresh window, exactly as it would run live).
Targets are 0 before SoccerNet's Goal timestamp (ball crosses the line) and 1 after it, so
reading the score ``d`` seconds after the goal traces speed vs accuracy with one model.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ..cards.model import CardNet
from ..cards.train import FRAME_SHAPE
from ..paths import MODELS
from .clips import CLIP_FPS, CLIP_FRAMES, PRE_S, load_clip

log = logging.getLogger(__name__)

DELAYS_S = (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
SELECT_DELAYS_S = (2.0, 3.0, 4.0)


@dataclass
class GoalTrainConfig:
    backbone: str = "regnet_y_800mf"
    frames: int = 40  # 3.2 s of context at 12.5 fps
    warmup: int = 10  # first frames of a window carry no loss: the GRU starts cold
    batch_size: int = 8
    epochs: int = 15
    patience: int = 4
    samples_per_epoch: int = 6000
    lr: float = 3e-4
    backbone_lr: float = 1e-4
    weight_decay: float = 1e-2
    goal_weight: float = 3.0  # sampling weight of goal clips
    num_workers: int = 12
    seed: int = 0


def _decode(jpegs: list[bytes]) -> np.ndarray:
    imgs = []
    for b in jpegs:
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        if img.shape[:2] != FRAME_SHAPE:
            img = cv2.resize(img, FRAME_SHAPE[::-1], interpolation=cv2.INTER_AREA)
        imgs.append(img[..., ::-1])
    return np.stack(imgs)


def frame_targets(goal_s: float, start_s: float, n: int = CLIP_FRAMES) -> np.ndarray:
    t = start_s + np.arange(n) / CLIP_FPS
    return (t >= goal_s).astype(np.float32) if np.isfinite(goal_s) else np.zeros(n, np.float32)


class GoalClips(Dataset):
    """Training: a random ``frames``-long window of a clip. Eval: the whole clip."""

    def __init__(self, index: pd.DataFrame, frames: int | None = 40, train: bool = True):
        self.index = index.reset_index(drop=True)
        self.frames, self.train = frames, train

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        r = self.index.iloc[i]
        jpegs = load_clip(r["clip_id"])
        y = frame_targets(r["goal_s"], r["start_s"], len(jpegs))
        if self.train:
            s = np.random.randint(0, len(jpegs) - self.frames + 1)
            jpegs, y = jpegs[s : s + self.frames], y[s : s + self.frames]
        x = _decode(jpegs).astype(np.float32)
        if self.train:
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1]
            x = np.clip((x - 128) * np.random.uniform(0.8, 1.2) + 128 + np.random.uniform(-20, 20), 0, 255)
        return torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2), torch.from_numpy(y)


@torch.no_grad()
def live_scores(model: CardNet, x: torch.Tensor, frames: int) -> torch.Tensor:
    """(B, T, 3, H, W) -> (B, T) live P(goal by t); NaN until a full window of context exists.

    The CNN runs once per frame; the GRU runs over the last ``frames`` frames for each t.
    """
    b, t = x.shape[:2]
    xf = (x.flatten(0, 1).float() - model.mean) / model.std
    f = model.proj(model.backbone(xf.contiguous(memory_format=torch.channels_last)).view(b, t, -1))
    win = f.unfold(1, frames, 1).permute(0, 1, 3, 2)  # (B, T - frames + 1, frames, D)
    h, _ = model.gru(win.flatten(0, 1))
    p = torch.sigmoid(model.head(h[:, -1]).float()).view(b, -1)
    return torch.cat([torch.full((b, frames - 1), float("nan"), device=p.device), p], 1)


@torch.no_grad()
def clip_live_scores(model: CardNet, index: pd.DataFrame, frames: int, batch_size: int = 4,
                     num_workers: int = 12) -> np.ndarray:
    """(N, CLIP_FRAMES) live scores for every clip in ``index``."""
    device = next(model.parameters()).device
    dl = DataLoader(GoalClips(index, None, train=False), batch_size=batch_size, num_workers=num_workers)
    model.eval()
    out = []
    for x, _ in dl:
        with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
            out.append(live_scores(model, x.to(device, non_blocking=True), frames).cpu().numpy())
    return np.concatenate(out)


def reference_times(index: pd.DataFrame, gap_s: float) -> np.ndarray:
    """Goal time for goal clips; for the rest, when the ball would have crossed (shot + median gap)."""
    g = index["goal_s"].to_numpy()
    return np.where(np.isfinite(g), g, index["anchor_s"].to_numpy() + gap_s)


def alert_scores(scores: np.ndarray, index: pd.DataFrame, gap_s: float, delays=DELAYS_S) -> dict[float, np.ndarray]:
    """{d: (N,) max live score from just before the shot until d s after the (reference) goal time}."""
    t = PRE_S - 0.5 + np.zeros(len(index))  # clip-relative start: 0.5 s before the first shot
    ref = reference_times(index, gap_s) - index["start_s"].to_numpy()
    out = {}
    for d in delays:
        lo = np.floor(t * CLIP_FPS).astype(int)
        hi = np.minimum(np.floor((ref + d) * CLIP_FPS).astype(int), scores.shape[1] - 1)
        out[d] = np.array([np.nanmax(s[a : b + 1]) if b >= a else 0.0 for s, a, b in zip(scores, lo, hi)])
    return out


def delay_report(scores: np.ndarray, index: pd.DataFrame, gap_s: float, thresholds: dict | None = None,
                 precisions=(0.7, 0.9)) -> tuple[pd.DataFrame, dict]:
    """AP of goal vs (all negatives / saved shots only) at each delay, plus recall at fixed precision.

    Thresholds for the recall columns are fitted here unless given (fit on valid, apply on test).
    """
    y = np.isfinite(index["goal_s"].to_numpy())
    on = (index["kind"] == "shot_on").to_numpy()
    fitted = {}
    rows = []
    for d, s in alert_scores(scores, index, gap_s).items():
        row = {"delay_s": d, "AP": average_precision_score(y, s),
               "AP_vs_saves": average_precision_score(y[y | on], s[y | on])}
        for p in precisions:
            th = (thresholds or {}).get((d, p))
            if th is None:
                th = _threshold_for_precision(y, s, p)
            fitted[(d, p)] = th
            fire = s >= th
            row[f"precision@p{p}"] = (fire & y).sum() / max(fire.sum(), 1)
            row[f"recall@p{p}"] = (fire & y).sum() / max(y.sum(), 1)
        rows.append(row)
    return pd.DataFrame(rows), fitted


def _threshold_for_precision(y: np.ndarray, s: np.ndarray, p: float) -> float:
    """Lowest threshold whose alerts (score >= threshold) reach precision ``p``."""
    order = np.argsort(-s, kind="stable")
    ss = s[order]
    prec = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
    last_of_tie = np.r_[ss[1:] != ss[:-1], True]  # firing at ss[i] fires every tied score
    ok = np.flatnonzero((prec >= p) & last_of_tie & (ss > 0))
    return float(ss[ok[-1]]) if len(ok) else float("inf")


def train(cfg: GoalTrainConfig, train_index: pd.DataFrame, valid_index: pd.DataFrame,
          out: Path = MODELS / "goal_vision.pt") -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    gap_s = float((train_index["goal_s"] - train_index["anchor_s"]).median())

    is_goal = np.isfinite(train_index["goal_s"].to_numpy())
    w = np.where(is_goal, cfg.goal_weight, 1.0)
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), cfg.samples_per_epoch, replacement=True)
    train_dl = DataLoader(GoalClips(train_index, cfg.frames, True), batch_size=cfg.batch_size, sampler=sampler,
                          num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True, drop_last=True)
    log.info("train clips %d (goals %d), valid clips %d (goals %d), shot->goal gap %.2fs", len(train_index),
             is_goal.sum(), len(valid_index), np.isfinite(valid_index["goal_s"]).sum(), gap_s)

    model = CardNet(cfg.backbone, n_classes=1, causal=True).to(device).to(memory_format=torch.channels_last)
    params = [{"params": model.backbone.parameters(), "lr": cfg.backbone_lr},
              {"params": [p for n, p in model.named_parameters() if not n.startswith("backbone.")], "lr": cfg.lr}]
    opt = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [cfg.backbone_lr, cfg.lr], total_steps=cfg.epochs * len(train_dl),
                                                pct_start=0.1)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best, best_state, best_epoch, history = -1.0, None, 0, []
    for epoch in range(1, cfg.epochs + 1):
        t0, total, n = time.time(), 0.0, 0
        model.train()
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x)[..., 0].float()
            loss = loss_fn(logits[:, cfg.warmup :], y[:, cfg.warmup :])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            total, n = total + loss.item() * len(x), n + len(x)
        rep, _ = delay_report(clip_live_scores(model, valid_index, cfg.frames, num_workers=cfg.num_workers),
                              valid_index, gap_s)
        score = float(rep[rep["delay_s"].isin(SELECT_DELAYS_S)]["AP"].mean())
        history.append({"epoch": epoch, "train_loss": total / n, **{f"AP@{d}s": a for d, a in zip(rep.delay_s, rep.AP)}})
        log.info("epoch %2d  train %.4f  valid AP by delay %s  (%.0fs)", epoch, total / n,
                 dict(zip(rep.delay_s, rep.AP.round(3))), time.time() - t0)
        if score > best:
            best, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= cfg.patience:
            log.info("early stop")
            break
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "config": asdict(cfg), "gap_s": gap_s, "valid_ap": best,
                "epoch": best_epoch, "history": history}, out)
    log.info("best valid AP %.4f (epoch %d) -> %s", best, best_epoch, out)
    return out


def load_goal_model(path: Path | str, device: str = "cuda") -> tuple[CardNet, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = CardNet(ckpt["config"]["backbone"], n_classes=1, pretrained=False, causal=True)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).to(memory_format=torch.channels_last).eval(), ckpt


def evaluate(model: CardNet, ckpt: dict, valid_index: pd.DataFrame, test_index: pd.DataFrame,
             num_workers: int = 12) -> dict[str, pd.DataFrame]:
    """Thresholds at fixed precision fitted on valid, everything reported on test."""
    frames, gap = ckpt["config"]["frames"], ckpt["gap_s"]
    sv = clip_live_scores(model, valid_index, frames, num_workers=num_workers)
    st = clip_live_scores(model, test_index, frames, num_workers=num_workers)
    _, th = delay_report(sv, valid_index, gap)
    rep, _ = delay_report(st, test_index, gap, thresholds=th)
    final = alert_scores(st, test_index, gap, delays=(2.0,))[2.0]
    by_kind = (test_index.assign(score_at_2s=final).groupby("kind")["score_at_2s"]
               .describe(percentiles=[0.5, 0.9])[["count", "mean", "50%", "90%"]].reset_index())
    return {"delays": rep, "by_kind": by_kind, "test_scores": pd.DataFrame(st, index=test_index["clip_id"])}
