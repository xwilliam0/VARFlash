"""Train / apply the visual card detector on pre-extracted clips."""

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
    num_workers: int = 12
    seed: int = 0


def clip_labels(events_json: str, window_s: float) -> np.ndarray:
    """(2,) 1 if a card of each class lies within ``window_s`` of the clip centre."""
    y = np.zeros(len(CARD_CLASSES), np.float32)
    for off, c in json.loads(events_json):
        if abs(off) <= window_s:
            y[c] = 1
    return y


class ClipDataset(Dataset):
    def __init__(self, index: pd.DataFrame, frames: int = 48, train: bool = True, jitter_s: float = 4.0,
                 radius: int = 4):
        self.index = index.reset_index(drop=True)
        self.frames, self.train, self.radius = frames, train, radius
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
            k = int(round(CLIP_FRAMES // 2 + off * CLIP_FPS)) - start
            y[max(k - self.radius, 0) : max(min(k + self.radius + 1, self.frames), 0), c] = 1
        return torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2), torch.from_numpy(y)


@torch.no_grad()
def clip_scores(model: CardNet, index: pd.DataFrame, frames: int = 48, window_s: float = 4.0,
                batch_size: int = 8, num_workers: int = 12) -> np.ndarray:
    """(N, 2) per-clip probabilities: max over frames within ``window_s`` of the clip centre."""
    device = next(model.parameters()).device
    dl = DataLoader(ClipDataset(index, frames, train=False), batch_size=batch_size, num_workers=num_workers)
    lo, hi = frames // 2 - int(window_s * CLIP_FPS), frames // 2 + int(window_s * CLIP_FPS) + 1
    model.eval()
    out = []
    for x, _ in dl:
        with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
            p = torch.sigmoid(model(x.to(device, non_blocking=True)).float())
        out.append(p[:, lo:hi].amax(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CARD_CLASSES)), np.float32)


def _clip_ap(model, index, cfg) -> dict[str, float]:
    p = clip_scores(model, index, cfg.frames, num_workers=cfg.num_workers)
    y = np.stack([clip_labels(e, 4.0) for e in index["events"]])
    return {c: float(average_precision_score(y[:, k], p[:, k])) if y[:, k].any() else float("nan")
            for k, c in enumerate(CARD_CLASSES)}


def train(cfg: CardTrainConfig, train_index: pd.DataFrame, valid_index: pd.DataFrame,
          out: Path = MODELS / "card_vision.pt") -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    labels = np.stack([clip_labels(e, cfg.jitter_s + cfg.frames / CLIP_FPS / 2) for e in train_index["events"]])
    w = np.where(labels[:, 1] > 0, cfg.red_weight, np.where(labels[:, 0] > 0, cfg.yellow_weight, 1.0))
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), cfg.samples_per_epoch, replacement=True)
    train_dl = DataLoader(ClipDataset(train_index, cfg.frames, True, cfg.jitter_s, cfg.radius), batch_size=cfg.batch_size,
                          sampler=sampler, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True,
                          drop_last=True)
    log.info("train clips %d (yellow %d, sending off %d), valid clips %d", len(train_index), int(labels[:, 0].sum()),
             int(labels[:, 1].sum()), len(valid_index))

    model = CardNet(cfg.backbone).to(device).to(memory_format=torch.channels_last)
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
                loss = loss_fn(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            total, n = total + loss.item() * len(x), n + len(x)
        ap = _clip_ap(model, valid_index, cfg)
        score = float(np.nanmean(list(ap.values())))
        history.append({"epoch": epoch, "train_loss": total / n, **ap})
        log.info("epoch %2d  train %.4f  valid clip AP %s  (%.0fs)", epoch, total / n,
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
    model = CardNet(ckpt["config"]["backbone"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).to(memory_format=torch.channels_last).eval(), ckpt
