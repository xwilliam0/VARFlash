"""Train / apply the visual offside verifier on pre-extracted clips.

Same network as the card verifier (ImageNet RegNetY-800MF fine-tuned end to end on full
16:9 frames + a BiGRU), one output: is there an offside call at this frame (+-1 s).
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ..cards.model import CardNet
from ..goals.train import _decode
from ..paths import MODELS
from .clips import CLIP_FPS, CLIP_FRAMES, PRE_S, load_clip

log = logging.getLogger(__name__)

CENTER = int(PRE_S * CLIP_FPS)  # clip frame of the candidate moment
SCORE_WINDOW_S = 4.0  # clip score = max frame score within this of the candidate


@dataclass
class OffsideTrainConfig:
    backbone: str = "regnet_y_800mf"
    frames: int = 48  # 12 s at 4 fps
    lead: int = 16  # frames before the candidate (4 s); the rest of the window is after it
    jitter_s: float = 3.0  # spotter candidates are a few seconds off; train for it
    radius: int = 4  # frames (+-1 s) around the offside count as positive
    batch_size: int = 6
    epochs: int = 12
    patience: int = 4
    samples_per_epoch: int = 6000
    lr: float = 3e-4
    backbone_lr: float = 1e-4
    weight_decay: float = 1e-2
    pos_weight: float = 3.0
    offside_weight: float = 2.0  # sampling weight of clips with an offside near the centre
    valid_candidates: int = 2500  # spotter candidates scored for model selection (plus every labelled clip)
    num_workers: int = 12
    seed: int = 0


def has_offside(offsides_json: str, window_s: float) -> bool:
    return any(abs(o) <= window_s for o in json.loads(offsides_json))


class OffsideClips(Dataset):
    def __init__(self, index: pd.DataFrame, cfg: OffsideTrainConfig, train: bool = True):
        self.index = index.reset_index(drop=True)
        self.cfg, self.train = cfg, train
        self.jitter = int(cfg.jitter_s * CLIP_FPS)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        r, cfg = self.index.iloc[i], self.cfg
        jpegs = load_clip(r["clip_id"])
        start = CENTER - cfg.lead
        if self.train:
            start += np.random.randint(-self.jitter, self.jitter + 1)
        start = int(np.clip(start, 0, CLIP_FRAMES - cfg.frames))
        x = _decode(jpegs[start : start + cfg.frames]).astype(np.float32)
        if self.train:
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1]
            x = np.clip((x - 128) * np.random.uniform(0.8, 1.2) + 128 + np.random.uniform(-20, 20), 0, 255)
        y = np.zeros((cfg.frames, 1), np.float32)
        for off in json.loads(r["offsides"]):
            k = int(round(CENTER + off * CLIP_FPS)) - start
            y[max(k - cfg.radius, 0) : max(min(k + cfg.radius + 1, cfg.frames), 0), 0] = 1
        return torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2), torch.from_numpy(y)


@torch.no_grad()
def clip_scores(model: CardNet, index: pd.DataFrame, cfg: OffsideTrainConfig, batch_size: int = 8) -> np.ndarray:
    """(N,) per-clip P(offside): max frame score within ``SCORE_WINDOW_S`` of the candidate."""
    device = next(model.parameters()).device
    dl = DataLoader(OffsideClips(index, cfg, train=False), batch_size=batch_size, num_workers=cfg.num_workers)
    w = int(SCORE_WINDOW_S * CLIP_FPS)
    lo, hi = max(cfg.lead - w, 0), cfg.lead + w + 1
    model.eval()
    out = []
    for x, _ in dl:
        with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
            p = torch.sigmoid(model(x.to(device, non_blocking=True)).float())
        out.append(p[:, lo:hi, 0].amax(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def train(cfg: OffsideTrainConfig, train_index: pd.DataFrame, valid_index: pd.DataFrame,
          out: Path = MODELS / "offside_vision.pt") -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    pos = np.array([has_offside(o, cfg.jitter_s + 2.0) for o in train_index["offsides"]])
    sampler = WeightedRandomSampler(torch.tensor(np.where(pos, cfg.offside_weight, 1.0), dtype=torch.double),
                                    cfg.samples_per_epoch, replacement=True)
    train_dl = DataLoader(OffsideClips(train_index, cfg, True), batch_size=cfg.batch_size, sampler=sampler,
                          num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True, drop_last=True)
    cand = valid_index["kind"] == "candidate"
    valid_index = pd.concat([valid_index[~cand], valid_index[cand].sample(min(cfg.valid_candidates, cand.sum()),
                                                                          random_state=0)])
    y_valid = np.array([has_offside(o, SCORE_WINDOW_S) for o in valid_index["offsides"]])
    log.info("train clips %d (offside %d), valid clips %d (offside %d)", len(train_index), pos.sum(),
             len(valid_index), y_valid.sum())

    model = CardNet(cfg.backbone, n_classes=1).to(device).to(memory_format=torch.channels_last)
    params = [{"params": model.backbone.parameters(), "lr": cfg.backbone_lr},
              {"params": [p for n, p in model.named_parameters() if not n.startswith("backbone.")], "lr": cfg.lr}]
    opt = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [cfg.backbone_lr, cfg.lr], total_steps=cfg.epochs * len(train_dl),
                                                pct_start=0.1)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg.pos_weight], device=device))

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
        ap = float(average_precision_score(y_valid, clip_scores(model, valid_index, cfg)))
        history.append({"epoch": epoch, "train_loss": total / n, "valid_clip_ap": ap})
        log.info("epoch %2d  train %.4f  valid clip AP %.4f  (%.0fs)", epoch, total / n, ap, time.time() - t0)
        if ap > best:
            best, best_epoch, best_state = ap, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= cfg.patience:
            log.info("early stop")
            break
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "config": asdict(cfg), "valid_clip_ap": best, "epoch": best_epoch,
                "history": history}, out)
    log.info("best valid clip AP %.4f (epoch %d) -> %s", best, best_epoch, out)
    return out


def load_offside_model(path: Path | str, device: str = "cuda") -> tuple[CardNet, OffsideTrainConfig]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = CardNet(ckpt["config"]["backbone"], n_classes=1, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).to(memory_format=torch.channels_last).eval(), OffsideTrainConfig(**ckpt["config"])
