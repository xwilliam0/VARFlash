"""Visual card detector: per-frame CNN (fine-tuned end to end) + temporal GRU.

Unlike the feature spotter, which sees frozen 512-d ImageNet features at 2 fps, this
model learns its own frame features at 4 fps from raw pixels, so it can pick up
small cues such as the colour of the card in the referee's hand.
"""

from __future__ import annotations

import torch
import torchvision
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    if name == "regnet_y_800mf":
        m = torchvision.models.regnet_y_800mf(weights="IMAGENET1K_V2" if pretrained else None)
        dim = m.fc.in_features
        m.fc = nn.Identity()
    elif name == "resnet18":
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        dim = m.fc.in_features
        m.fc = nn.Identity()
    else:
        raise ValueError(name)
    return m, dim


class CardNet(nn.Module):
    """(B, T, 3, H, W) uint8-range RGB frames -> (B, T, n_classes) logits."""

    def __init__(self, backbone: str = "regnet_y_800mf", n_classes: int = 2, dim: int = 256,
                 dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        self.backbone, feat = _backbone(backbone, pretrained)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) * 255)
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat, dim), nn.ReLU())
        self.gru = nn.GRU(dim, dim // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        b, t = x.shape[:2]
        x = (x.flatten(0, 1).float() - self.mean) / self.std
        f = self.backbone(x.contiguous(memory_format=torch.channels_last)).view(b, t, -1)
        h, _ = self.gru(self.proj(f))
        return self.head(h)
