"""SoccerNet-compatible frame features for new matches: ResNet-152 (Keras weights) + PCA-512.

SoccerNet's ``ResNET_TF2_PCA512`` features come from Keras' ImageNet ResNet-152
(``avg_pool`` output, 2048-d) on 2 fps centre-cropped 224x224 frames, then a PCA to
512-d. This module re-implements that backbone in PyTorch (so it runs on any GPU
without TensorFlow), loads the original Keras ``.h5`` weights, and recovers the PCA
as an affine map fitted on SoccerNet's own (raw 2048-d, PCA-512) feature pairs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from ..paths import MODELS, SOCCERNET_DATA
from .data import FEATURE_FILE, FPS

log = logging.getLogger(__name__)

KERAS_WEIGHTS_URL = (
    "https://storage.googleapis.com/tensorflow/keras-applications/resnet/"
    "resnet152_weights_tf_dim_ordering_tf_kernels.h5"
)
KERAS_WEIGHTS = MODELS / "backbones" / "resnet152_weights_tf_dim_ordering_tf_kernels.h5"
PCA_FILE = MODELS / "backbones" / "soccernet_pca512.npz"
RAW_FILE = "{half}_ResNET_TF2.npy"
CAFFE_MEAN_BGR = (103.939, 116.779, 123.68)
BN_EPS = 1.001e-5
RESNET152_BLOCKS = (3, 8, 36, 3)


# ---- backbone ---------------------------------------------------------------


def _bn(c: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(c, eps=BN_EPS)


class KerasBottleneck(nn.Module):
    """Keras ResNet v1 bottleneck: the stride sits on the first 1x1 conv (not the 3x3)."""

    def __init__(self, cin: int, filters: int, stride: int, shortcut_conv: bool):
        super().__init__()
        cout = 4 * filters
        self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride), _bn(cout)) if shortcut_conv else nn.Identity()
        self.conv1, self.bn1 = nn.Conv2d(cin, filters, 1, stride), _bn(filters)
        self.conv2, self.bn2 = nn.Conv2d(filters, filters, 3, 1, padding=1), _bn(filters)
        self.conv3, self.bn3 = nn.Conv2d(filters, cout, 1), _bn(cout)

    def forward(self, x):
        h = torch.relu(self.bn1(self.conv1(x)))
        h = torch.relu(self.bn2(self.conv2(h)))
        return torch.relu(self.shortcut(x) + self.bn3(self.conv3(h)))


class KerasResNet152(nn.Module):
    """Keras ``ResNet152`` up to ``avg_pool``: (B, 3, 224, 224) caffe-preprocessed -> (B, 2048)."""

    def __init__(self):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(3, 64, 7, 2, padding=3), _bn(64)
        self.pool = nn.Sequential(nn.ZeroPad2d(1), nn.MaxPool2d(3, 2))
        stages, cin = [], 64
        for s, (n, filters) in enumerate(zip(RESNET152_BLOCKS, (64, 128, 256, 512))):
            blocks = []
            for b in range(n):
                blocks.append(KerasBottleneck(cin, filters, (1 if s == 0 else 2) if b == 0 else 1, b == 0))
                cin = 4 * filters
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        return self.stages(x).mean(dim=(2, 3))


def _keras_layer_map(model: KerasResNet152) -> dict[str, nn.Module]:
    out = {"conv1_conv": model.conv1, "conv1_bn": model.bn1}
    for s, stage in enumerate(model.stages):
        for b, blk in enumerate(stage):
            p = f"conv{s + 2}_block{b + 1}_"
            out.update({p + "1_conv": blk.conv1, p + "1_bn": blk.bn1, p + "2_conv": blk.conv2,
                        p + "2_bn": blk.bn2, p + "3_conv": blk.conv3, p + "3_bn": blk.bn3})
            if b == 0:
                out.update({p + "0_conv": blk.shortcut[0], p + "0_bn": blk.shortcut[1]})
    return out


def load_keras_resnet152(h5_path: Path = KERAS_WEIGHTS) -> KerasResNet152:
    import h5py

    if not h5_path.exists():
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading Keras ResNet-152 weights -> %s", h5_path)
        torch.hub.download_url_to_file(KERAS_WEIGHTS_URL, str(h5_path))
    model = KerasResNet152()
    layers = _keras_layer_map(model)
    with h5py.File(h5_path, "r") as f:
        for name, mod in layers.items():
            keys = [k.decode() if hasattr(k, "decode") else k for k in f[name].attrs["weight_names"]]
            w = {k.split("/")[-1].split(":")[0]: np.asarray(f[name][k]) for k in keys}
            with torch.no_grad():
                if isinstance(mod, nn.Conv2d):
                    mod.weight.copy_(torch.from_numpy(w["kernel"].transpose(3, 2, 0, 1)))  # HWIO -> OIHW
                    mod.bias.copy_(torch.from_numpy(w["bias"]))
                else:
                    mod.weight.copy_(torch.from_numpy(w["gamma"]))
                    mod.bias.copy_(torch.from_numpy(w["beta"]))
                    mod.running_mean.copy_(torch.from_numpy(w["moving_mean"]))
                    mod.running_var.copy_(torch.from_numpy(w["moving_variance"]))
    return model.eval()


# ---- frames -------------------------------------------------------------------


def preprocess(frames: np.ndarray, swap_rb: bool = True) -> torch.Tensor:
    """uint8 (N, 224, 224, 3) OpenCV BGR frames -> caffe-style float tensor (N, 3, 224, 224).

    SoccerNet passed OpenCV's BGR frames to Keras' ``preprocess_input``, which assumes
    RGB and flips channels, so the network effectively saw RGB minus the BGR means.
    ``swap_rb=True`` reproduces that quirk (the released features were made this way).
    """
    x = torch.from_numpy(frames).float()
    if swap_rb:
        x = x.flip(-1)
    x = x - torch.tensor(CAFFE_MEAN_BGR)
    return x.permute(0, 3, 1, 2).contiguous()


def _crop(frame: np.ndarray, transform: str) -> np.ndarray:
    h, w = frame.shape[:2]
    if transform == "resize":
        return cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
    if transform != "crop":
        raise ValueError(transform)
    if h != 224:
        frame = cv2.resize(frame, (round(w * 224 / h), 224), interpolation=cv2.INTER_AREA)
    off = (frame.shape[1] - 224) // 2
    return frame[:, off : off + 224]


def iter_frames(video: Path | str, fps: float = FPS, transform: str = "crop", start_s: float = 0.0,
                duration_s: float | None = None, batch: int = 256):
    """Yield uint8 batches (B, 224, 224, 3) sampled at ``fps`` like SoccerNet's ``FrameCV``.

    FrameCV keeps 1-indexed frame i when ``i % (video_fps / fps) < 1``, i.e. the first
    sample is ~0.5 s into the video at 2 fps; we keep the same grid so features line up.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = vfps / fps
    first, last = start_s * vfps, None if duration_s is None else (start_s + duration_s) * vfps
    if start_s > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(first) - 1, 0))
    i = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    buf = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i < first:
            continue
        if last is not None and i > last:
            break
        if i % step < 1:
            buf.append(_crop(frame, transform))
            if len(buf) == batch:
                yield np.stack(buf)
                buf = []
    if buf:
        yield np.stack(buf)
    cap.release()


# ---- PCA ------------------------------------------------------------------------


def fit_pca(root: Path = SOCCERNET_DATA, max_halves: int = 20, out: Path = PCA_FILE) -> dict:
    """Recover SoccerNet's PCA-512 as ``pca = raw @ W + b`` from paired feature files.

    PCA is an affine map, so least squares on a few thousand pairs recovers it exactly
    (residual ~ float32 noise). Saves ``W`` (2048, 512) and ``b`` (512,).
    """
    xs, ys = [], []
    for raw in sorted(root.rglob(RAW_FILE.format(half="*")))[:max_halves]:
        pca = raw.with_name(raw.name.replace("ResNET_TF2", "ResNET_TF2_PCA512"))
        if pca.exists():
            x, y = np.load(raw), np.load(pca)
            n = min(len(x), len(y))
            xs.append(x[:n])
            ys.append(y[:n])
    if not xs:
        raise RuntimeError("no (ResNET_TF2, ResNET_TF2_PCA512) pairs found; download 1_ResNET_TF2.npy first")
    X, Y = np.concatenate(xs).astype(np.float64), np.concatenate(ys).astype(np.float64)
    A = np.hstack([X, np.ones((len(X), 1))])
    sol, *_ = np.linalg.lstsq(A, Y, rcond=None)
    W, b = sol[:-1].astype(np.float32), sol[-1].astype(np.float32)
    rel = float(np.linalg.norm(A @ sol - Y) / np.linalg.norm(Y))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, W=W, b=b)
    log.info("PCA fitted on %d frames from %d halves, relative residual %.2e -> %s", len(X), len(xs), rel, out)
    return {"frames": len(X), "halves": len(xs), "rel_residual": rel}


def load_pca(path: Path = PCA_FILE) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `varflash spotting fit-pca` first")
    d = np.load(path)
    return d["W"], d["b"]


# ---- extraction -----------------------------------------------------------------


class FeatureExtractor:
    """Video file -> (N, 2048) raw and (N, 512) PCA features at 2 fps."""

    def __init__(self, device: str | None = None, transform: str = "crop", swap_rb: bool = True,
                 pca_path: Path = PCA_FILE, half_precision: bool = False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # TF32 convolutions drift ~1e-3 from Keras' float32 output; keep exact parity.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.model = load_keras_resnet152().to(self.device)
        self.dtype = torch.float16 if half_precision and self.device.startswith("cuda") else torch.float32
        self.model = self.model.to(self.dtype)
        self.transform, self.swap_rb = transform, swap_rb
        self.pca = load_pca(pca_path) if pca_path is not None and Path(pca_path).exists() else None

    @torch.no_grad()
    def raw(self, video: Path | str, **kw) -> np.ndarray:
        out = []
        for frames in iter_frames(video, transform=self.transform, **kw):
            x = preprocess(frames, self.swap_rb).to(self.device, self.dtype)
            out.append(self.model(x).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 2048), np.float32)

    def to_pca(self, raw: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("no PCA loaded; run `varflash spotting fit-pca` first")
        W, b = self.pca
        return (raw @ W + b).astype(np.float32)

    def extract_match(self, videos: list[Path], out_dir: Path, keep_raw: bool = False, **kw) -> list[Path]:
        """One video per half -> ``{1,2}_ResNET_TF2_PCA512.npy`` in ``out_dir`` (SoccerNet layout)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for half, video in enumerate(videos, 1):
            raw = self.raw(video, **kw)
            if keep_raw:
                np.save(out_dir / RAW_FILE.format(half=half), raw)
            path = out_dir / FEATURE_FILE.format(half=half)
            np.save(path, self.to_pca(raw))
            log.info("half %d: %d frames (%.1f min) -> %s", half, len(raw), len(raw) / FPS / 60, path)
            written.append(path)
        return written
