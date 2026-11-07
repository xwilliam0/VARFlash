"""Train / apply the visual card detector on pre-extracted clips.

``causal=True`` trains a past-only verifier for live alerts: a forward-only GRU whose
targets sit ``delay_s`` after the card, so the alert can fire that long after the card
is shown using no later video. It is scored the way it would run live, at several
times after the card (``live_report``).
"""

from __future__ import annotations

import copy
import json
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

from ..paths import MODELS
from .clips import CARD_CLASSES, CLIP_FPS, CLIP_FRAMES, load_clip
from .model import CardNet

log = logging.getLogger(__name__)

FRAME_SHAPE = (224, 398)


@dataclass
class CardTrainConfig:
    backbone: str = "regnet_y_800mf"
    frames: int = 48  # 12 s at 4 fps
    jitter_s: float = 4.0  # spotter candidates are a few seconds off; train for it
    radius: int = 4  # frames (+-1 s) around the card count as positive
    batch_size: int = 6
    epochs: int = 12
    patience: int = 4
    samples_per_epoch: int = 6000
    lr: float = 3e-4
    backbone_lr: float = 1e-4
    weight_decay: float = 1e-2
    pos_weight: float = 5.0
    red_weight: float = 15.0  # sampling weight of clips with a sending-off
    yellow_weight: float = 2.0
    causal: bool = False
    delay_s: float = 0.0  # causal only: targets sit this long after the card
    warmup: int = 8  # causal only: first frames of a window carry no loss, the GRU starts cold
    num_workers: int = 12
    seed: int = 0

    @property
    def delay(self) -> int:
        return int(round(self.delay_s * CLIP_FPS))


LIVE_DELAYS_S = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
LIVE_WINDOW_S = 4.0  # a clip is positive if a card is within this of its centre


def clip_labels(events_json: str, window_s: float) -> np.ndarray:
    """(2,) 1 if a card of each class lies within ``window_s`` of the clip centre."""
    y = np.zeros(len(CARD_CLASSES), np.float32)
    for off, c in json.loads(events_json):
        if abs(off) <= window_s:
            y[c] = 1
    return y


class ClipDataset(Dataset):
    def __init__(self, index: pd.DataFrame, frames: int = 48, train: bool = True, jitter_s: float = 4.0,
                 radius: int = 4, delay: int = 0):
        self.index = index.reset_index(drop=True)
        self.frames, self.train, self.radius, self.delay = frames, train, radius, delay
        self.jitter = int(jitter_s * CLIP_FPS)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        r = self.index.iloc[i]
        jpegs = load_clip(r["clip_id"])
        start = CLIP_FRAMES // 2 - self.frames // 2
        if self.train:
            start = int(np.clip(start + np.random.randint(-self.jitter, self.jitter + 1), 0, CLIP_FRAMES - self.frames))
        imgs = []
        for b in jpegs[start : start + self.frames]:
            img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            if img.shape[:2] != FRAME_SHAPE:
                img = cv2.resize(img, FRAME_SHAPE[::-1], interpolation=cv2.INTER_AREA)
            imgs.append(img[..., ::-1])  # BGR -> RGB
        x = np.stack(imgs).astype(np.float32)
        if self.train:
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1]
            # Brightness/contrast only: hue jitter would blur the yellow/red distinction.
            x = np.clip((x - 128) * np.random.uniform(0.8, 1.2) + 128 + np.random.uniform(-20, 20), 0, 255)
        y = np.zeros((self.frames, len(CARD_CLASSES)), np.float32)
        for off, c in json.loads(r["events"]):
            k = int(round(CLIP_FRAMES // 2 + off * CLIP_FPS)) - start + self.delay
            y[max(k - self.radius, 0) : max(min(k + self.radius + 1, self.frames), 0), c] = 1
        return torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2), torch.from_numpy(y)


@torch.no_grad()
def live_scores(model: CardNet, x: torch.Tensor, frames: int) -> torch.Tensor:
    """(B, T, 3, H, W) -> (B, T, C) probabilities exactly as a causal model emits them live.

    Frame t sees frames ``max(0, t - frames + 1) .. t`` through a cold-started GRU, the
    same context length it was trained on. The CNN runs once per frame.
    """
    b, t = x.shape[:2]
    xf = (x.flatten(0, 1).float() - model.mean) / model.std
    f = model.proj(model.backbone(xf.contiguous(memory_format=torch.channels_last)).view(b, t, -1))
    w = min(frames, t)
    h, _ = model.gru(f[:, :w])
    out = [model.head(h)]
    if t > w:
        win = f.unfold(1, w, 1).permute(0, 1, 3, 2)[:, 1:]  # (B, t - w, w, D): windows ending at w .. t-1
        h, _ = model.gru(win.flatten(0, 1))
        out.append(model.head(h[:, -1]).view(b, t - w, -1))
    return torch.sigmoid(torch.cat(out, 1).float())


@torch.no_grad()
def clip_live(model: CardNet, index: pd.DataFrame, frames: int, batch_size: int = 4,
              num_workers: int = 12, tta: bool = False) -> np.ndarray:
    """(N, CLIP_FRAMES, C) live scores over each whole stored clip (``tta``: averaged with the mirror image)."""
    device = next(model.parameters()).device
    dl = DataLoader(ClipDataset(index, CLIP_FRAMES, train=False), batch_size=batch_size, num_workers=num_workers)
    model.eval()
    out = []
    for x, _ in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
            p = live_scores(model, x, frames)
            if tta:
                p = (p + live_scores(model, x.flip(-1), frames)) / 2
        out.append(p.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, CLIP_FRAMES, len(CARD_CLASSES)), np.float32)


@torch.no_grad()
def clip_scores(model: CardNet, index: pd.DataFrame, frames: int = 48, window_s: float = 4.0,
                batch_size: int = 8, num_workers: int = 12, delay: int = 0, tta: bool = False) -> np.ndarray:
    """(N, 2) per-clip probabilities: max over frames within ``window_s`` of the clip centre.

    A causal model is run live over the whole clip and its scores are shifted back by
    ``delay`` frames, so frame t scores a card shown at t (same as the spotter's ``event_scores``).
    ``tta`` averages each frame's probability with that of the mirrored clip (training used hflip).
    """
    if model.causal:
        live = clip_live(model, index, frames, max(batch_size // 2, 1), num_workers, tta)
        ev = np.zeros_like(live)
        ev[:, : CLIP_FRAMES - delay] = live[:, delay:]
        c = CLIP_FRAMES // 2
        return ev[:, c - int(window_s * CLIP_FPS) : c + int(window_s * CLIP_FPS) + 1].max(1)
    device = next(model.parameters()).device
    dl = DataLoader(ClipDataset(index, frames, train=False), batch_size=batch_size, num_workers=num_workers)
    lo, hi = frames // 2 - int(window_s * CLIP_FPS), frames // 2 + int(window_s * CLIP_FPS) + 1
    model.eval()
    out = []
    for x, _ in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
            p = torch.sigmoid(model(x).float())
            if tta:
                p = (p + torch.sigmoid(model(x.flip(-1)).float())) / 2
        out.append(p[:, lo:hi].amax(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CARD_CLASSES)), np.float32)


def card_offsets(index: pd.DataFrame, window_s: float = LIVE_WINDOW_S) -> np.ndarray:
    """(N,) seconds from the clip centre to the nearest card within ``window_s`` (0 if none)."""
    out = np.zeros(len(index))
    for i, e in enumerate(index["events"]):
        offs = [off for off, _ in json.loads(e) if abs(off) <= window_s]
        if offs:
            out[i] = min(offs, key=abs)
    return out


def live_alert_scores(live: np.ndarray, index: pd.DataFrame, delays_s=LIVE_DELAYS_S,
                      window_s: float = LIVE_WINDOW_S) -> dict[float, np.ndarray]:
    """{d: (N, C) highest live score from ``window_s`` before the clip centre until d s after the card}.

    Clips without a card use the centre as the reference. Every score counted here was
    available live by then, so this is the alert a live system could have raised.
    """
    c = CLIP_FRAMES // 2
    lo = c - int(window_s * CLIP_FPS)
    ref = c + np.round(card_offsets(index, window_s) * CLIP_FPS).astype(int)
    out = {}
    for d in delays_s:
        hi = np.minimum(ref + int(round(d * CLIP_FPS)), CLIP_FRAMES - 1)
        out[d] = np.stack([s[lo : h + 1].max(0) for s, h in zip(live, hi)]) if len(live) else np.zeros((0, live.shape[-1]))
    return out


def live_report(valid_live: np.ndarray, valid_index: pd.DataFrame, test_live: np.ndarray, test_index: pd.DataFrame,
                precision: float = 0.5) -> pd.DataFrame:
    """AP by seconds after the card, and test recall at a threshold set for ``precision`` on valid."""
    from ..goals.train import _threshold_for_precision

    yv = np.stack([clip_labels(e, LIVE_WINDOW_S) for e in valid_index["events"]])
    yt = np.stack([clip_labels(e, LIVE_WINDOW_S) for e in test_index["events"]])
    sv, st = live_alert_scores(valid_live, valid_index), live_alert_scores(test_live, test_index)
    rows = []
    for d in LIVE_DELAYS_S:
        for k, c in enumerate(CARD_CLASSES):
            th = _threshold_for_precision(yv[:, k].astype(bool), sv[d][:, k], precision)
            fire = st[d][:, k] >= th
            pos = yt[:, k].astype(bool)
            rows.append({"after_card_s": d, "class": c,
                         "valid_AP": average_precision_score(yv[:, k], sv[d][:, k]) if pos.any() else float("nan"),
                         "test_AP": average_precision_score(yt[:, k], st[d][:, k]) if pos.any() else float("nan"),
                         f"test_precision@p{precision}": (fire & pos).sum() / max(fire.sum(), 1),
                         f"test_recall@p{precision}": (fire & pos).sum() / max(pos.sum(), 1)})
    return pd.DataFrame(rows)


def _clip_ap(model, index, cfg) -> dict[str, float]:
    """Validation score used to pick the epoch. Causal: live AP ``delay_s`` after the card."""
    y = np.stack([clip_labels(e, LIVE_WINDOW_S) for e in index["events"]])
    if cfg.causal:
        live = clip_live(model, index, cfg.frames, num_workers=cfg.num_workers)
        p = live_alert_scores(live, index, (cfg.delay_s,))[cfg.delay_s]
    else:
        p = clip_scores(model, index, cfg.frames, num_workers=cfg.num_workers)
    return {c: float(average_precision_score(y[:, k], p[:, k])) if y[:, k].any() else float("nan")
            for k, c in enumerate(CARD_CLASSES)}


def train(cfg: CardTrainConfig, train_index: pd.DataFrame, valid_index: pd.DataFrame,
          out: Path = MODELS / "card_vision.pt") -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if cfg.delay_s and not cfg.causal:
        raise ValueError("delay_s only makes sense with causal=True")

    labels = np.stack([clip_labels(e, cfg.jitter_s + cfg.frames / CLIP_FPS / 2) for e in train_index["events"]])
    w = np.where(labels[:, 1] > 0, cfg.red_weight, np.where(labels[:, 0] > 0, cfg.yellow_weight, 1.0))
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), cfg.samples_per_epoch, replacement=True)
    train_dl = DataLoader(ClipDataset(train_index, cfg.frames, True, cfg.jitter_s, cfg.radius, cfg.delay),
                          batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers, pin_memory=True,
                          persistent_workers=True, drop_last=True)
    log.info("train clips %d (yellow %d, sending off %d), valid clips %d, causal %s, delay %.1fs", len(train_index),
             int(labels[:, 0].sum()), int(labels[:, 1].sum()), len(valid_index), cfg.causal, cfg.delay_s)

    skip = cfg.warmup if cfg.causal else 0
    model = CardNet(cfg.backbone, causal=cfg.causal).to(device).to(memory_format=torch.channels_last)
    params = [{"params": model.backbone.parameters(), "lr": cfg.backbone_lr},
              {"params": [p for n, p in model.named_parameters() if not n.startswith("backbone.")], "lr": cfg.lr}]
    opt = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [cfg.backbone_lr, cfg.lr], total_steps=steps, pct_start=0.1)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.full((len(CARD_CLASSES),), cfg.pos_weight, device=device))

    best, best_state, best_epoch, history = -1.0, None, 0, []
    for epoch in range(1, cfg.epochs + 1):
        t0, total, n = time.time(), 0.0, 0
        model.train()
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
                loss = loss_fn(model(x)[:, skip:].float(), y[:, skip:])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            total, n = total + loss.item() * len(x), n + len(x)
        ap = _clip_ap(model, valid_index, cfg)
        score = float(np.nanmean(list(ap.values())))
        history.append({"epoch": epoch, "train_loss": total / n, **ap})
        log.info("epoch %2d  train %.4f  valid %s AP %s  (%.0fs)", epoch, total / n,
                 f"live@{cfg.delay_s:g}s" if cfg.causal else "clip",
                 {k: round(v, 3) for k, v in ap.items()}, time.time() - t0)
        if score > best:
            best, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= cfg.patience:
            log.info("early stop")
            break
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "config": asdict(cfg), "valid_clip_ap": best, "epoch": best_epoch,
                "history": history}, out)
    log.info("best valid clip AP %.4f (epoch %d) -> %s", best, best_epoch, out)
    return out


def load_card_model(path: Path | str, device: str = "cuda") -> tuple[CardNet, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = CardNet(ckpt["config"]["backbone"], pretrained=False, causal=ckpt["config"].get("causal", False))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).to(memory_format=torch.channels_last).eval(), ckpt
