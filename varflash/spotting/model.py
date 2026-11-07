"""Temporal action-spotting network over per-frame video features.

The input is not video pixels but a sequence of precomputed frame embeddings
(SoccerNet's 512-d ResNet-152 + PCA features at 2 fps, see ``features.py``).
This model only has to learn *when* things happen from how those embeddings
change over time; the frozen ImageNet backbone already handled *what* is in
each frame.

Shape conventions: B = batch, T = frames (2 per second), D = hidden width,
C = event classes.
"""

from __future__ import annotations

import torch
from torch import nn


class DilatedBlock(nn.Module):
    """Residual 1D convolution over time: ``x + dropout(relu(bn(conv(x))))``.

    Kernel size 3 with dilation ``d`` looks at frames ``t - d, t, t + d`` (or
    ``t - 2d, t - d, t`` when causal), so stacking blocks with doubling
    dilations grows the receptive field exponentially at constant cost.
    Padding keeps the output length equal to T.
    """

    def __init__(self, dim: int, dilation: int, dropout: float, causal: bool = False):
        super().__init__()
        # Causal: pad only on the left so frame t never sees frames > t.
        self.pad = (2 * dilation, 0) if causal else (dilation, dilation)
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, dilation=dilation)
        self.norm = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # (B, D, T)
        return x + self.drop(torch.relu(self.norm(self.conv(nn.functional.pad(x, self.pad)))))


class TemporalSpotter(nn.Module):
    """Per-frame multi-label event logits from a sequence of frame features.

    Dilated convolutions see ~±15 s of context cheaply; a BiGRU on top lets
    each frame use the whole window (e.g. a celebration after a goal, or the
    referee's VAR signal after an offside check).

    ``causal=True`` makes every output depend only on past frames (left-padded
    convolutions, forward-only GRU), so it can run live on a broadcast; train it
    with delayed targets to give it a fixed look-ahead budget.
    """

    def __init__(self, in_dim: int = 512, dim: int = 128, n_classes: int = 17, dilations=(1, 2, 4, 8, 16),
                 dropout: float = 0.2, causal: bool = False):
        super().__init__()
        self.causal = causal
        # Compress the 512-d backbone features to a smaller working width.
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, dim), nn.ReLU())
        # Each block adds 2 * dilation frames of context: 1 + 2 * (1+2+4+8+16) = 63
        # frames, i.e. ~31 s at 2 fps (±15 s centred, or the past 31 s when causal).
        self.tcn = nn.Sequential(*[DilatedBlock(dim, d, dropout, causal) for d in dilations])
        # Bidirectional: two directions of dim // 2 concatenate back to dim.
        # Causal: one forward direction of the full dim, so it never reads the future.
        self.gru = nn.GRU(dim, dim if causal else dim // 2, batch_first=True, bidirectional=not causal)
        # Independent logit per class per frame (multi-label: use sigmoid, not softmax).
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):  # (B, T, in_dim) -> (B, T, C)
        h = self.proj(x).transpose(1, 2)  # (B, D, T): Conv1d wants channels before time
        h = self.tcn(h).transpose(1, 2)  # back to (B, T, D) for the GRU
        h, _ = self.gru(h)  # (B, T, D)
        return self.head(h)
