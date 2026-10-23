"""Temporal action-spotting network over per-frame video features."""

from __future__ import annotations

import torch
from torch import nn


class DilatedBlock(nn.Module):
    def __init__(self, dim: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # (B, D, T)
        return x + self.drop(torch.relu(self.norm(self.conv(x))))


class TemporalSpotter(nn.Module):
    """Per-frame multi-label event logits from a sequence of frame features.

    Dilated convolutions see ~±15 s of context cheaply; a BiGRU on top lets
    each frame use the whole window (e.g. a celebration after a goal, or the
    referee's VAR signal after an offside check).
    """

    def __init__(self, in_dim: int = 512, dim: int = 128, n_classes: int = 17, dilations=(1, 2, 4, 8, 16), dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, dim), nn.ReLU())
        self.tcn = nn.Sequential(*[DilatedBlock(dim, d, dropout) for d in dilations])
        self.gru = nn.GRU(dim, dim // 2, batch_first=True, bidirectional=True)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):  # (B, T, in_dim) -> (B, T, C)
        h = self.proj(x).transpose(1, 2)
        h = self.tcn(h).transpose(1, 2)
        h, _ = self.gru(h)
        return self.head(h)
