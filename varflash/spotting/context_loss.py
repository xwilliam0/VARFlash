"""Context-aware segmentation loss for action spotting.

Reimplementation of the temporal-segmentation term in Cioppa et al.,
"A Context-Aware Loss Function for Action Spotting in Soccer Videos"
(CVPR 2020, arXiv:1912.01326). Their full network also regresses timestamps
with a separate spotting head; this module is only the per-frame loss, so it
drops onto the existing dense spotter and the public SoccerNet features.

Around each event the loss treats six zones (frames at 2 fps, class-specific):

- far before / far after: the frame must not fire
- transition into the event: that penalty fades to zero
- just before: no loss (a promising attack may or may not become a goal)
- the spot: the score must be high; that requirement fades out over the frames just after
- after that, a transition back to "must not fire"

A margin (tau) zeroes the loss once a score is already good enough, so
training spends its gradient on the frames that are still wrong.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

# (K1, K2, K3, K4) in frames. K1 < K2 < 0 < K3 < K4.
# Goal, card and substitution are the ResNet slices from the paper (their
# seconds times 2 fps). SoccerNet-v2's other classes are not in that paper;
# those slices are shorter so a throw-in does not blank the next minute.
SLICES: dict[str, tuple[int, int, int, int]] = {
    "Goal": (-40, -20, 120, 180),
    "Penalty": (-40, -20, 40, 80),
    "Substitution": (-80, -40, 20, 40),
    "Yellow card": (-40, -20, 20, 40),
    "Red card": (-40, -20, 20, 40),
    "Yellow->red card": (-40, -20, 20, 40),
    "Sending off": (-40, -20, 20, 40),
    "Offside": (-40, -20, 20, 40),
    "Foul": (-16, -6, 12, 24),
    "Kick-off": (-12, -4, 8, 16),
    "Shots on target": (-12, -4, 8, 16),
    "Shots off target": (-12, -4, 8, 16),
    "Clearance": (-12, -4, 8, 16),
    "Ball out of play": (-12, -4, 8, 16),
    "Throw-in": (-12, -4, 8, 16),
    "Indirect free-kick": (-12, -4, 8, 16),
    "Direct free-kick": (-12, -4, 8, 16),
    "Corner": (-12, -4, 8, 16),
}
_BRIEF = (-12, -4, 8, 16)


def bounds_for(class_names: list[str]) -> np.ndarray:
    """(C, 4) int32 slicing parameters, one row per model class."""
    rows = [SLICES.get(name, _BRIEF) for name in class_names]
    bounds = np.asarray(rows, np.int32)
    if bounds.shape != (len(class_names), 4):
        raise ValueError(class_names)
    if not np.all((bounds[:, 0] < bounds[:, 1]) & (bounds[:, 1] < 0) & (bounds[:, 2] > 0) & (bounds[:, 2] < bounds[:, 3])):
        raise ValueError(f"slices must satisfy K1 < K2 < 0 < K3 < K4, got {bounds}")
    return bounds


def time_shifts(n_frames: int, events: list[tuple[int, int]], bounds: np.ndarray, delay: int = 0) -> np.ndarray:
    """(n_frames, C) int32 time-shift of each frame from its dominant event.

    ``s = frame - event_frame`` after applying ``delay`` (same shift as
    ``frame_targets``). A class with no events is stored as K1, which the loss
    treats as far before. When a frame sits in both events' transition zones,
    the nearer event wins, and an exact tie goes to the one that already happened.
    """
    bounds = np.asarray(bounds, np.int32)
    n_classes = bounds.shape[0]
    out = np.broadcast_to(bounds[:, 0], (n_frames, n_classes)).copy()
    if n_frames == 0:
        return out
    frames = np.arange(n_frames)
    by_class: list[list[int]] = [[] for _ in range(n_classes)]
    for frame, cls in events:
        if 0 <= cls < n_classes:
            by_class[cls].append(int(frame) + delay)
    for cls, times in enumerate(by_class):
        if not times:
            continue
        ev = np.unique(np.asarray(times, np.int32))
        # idx = first event strictly after this frame, so the event on this frame is "past".
        idx = np.searchsorted(ev, frames, side="right")
        has_past = idx > 0
        has_fut = idx < len(ev)
        s_past = np.where(has_past, frames - ev[np.clip(idx - 1, 0, None)], 0)
        s_fut = np.where(has_fut, frames - ev[np.clip(idx, 0, len(ev) - 1)], 0)
        k1, k2, k3, k4 = (int(v) for v in bounds[cls])
        just_after = has_past & (s_past >= 0) & (s_past < k3)
        trans_after = has_past & (s_past >= k3) & (s_past < k4)
        far_before = ~has_fut | (s_fut <= k1)
        trans_before = has_fut & (s_fut > k1) & (s_fut <= k2)
        use_past = just_after | (trans_after & far_before) | (trans_after & trans_before & (s_past <= -s_fut))
        use_fut = ~use_past & has_fut
        use_past = use_past | (~use_fut & has_past)
        out[:, cls] = np.where(use_past, s_past, np.where(use_fut, s_fut, k1))
    return out


class ContextAwareLoss(nn.Module):
    """Mean context-aware loss. ``shifts`` is the int encoding from ``time_shifts``."""

    def __init__(self, bounds: np.ndarray, tau_max: float = 0.9, tau_min: float = 0.1):
        super().__init__()
        if not 0.0 < tau_min < tau_max < 1.0:
            raise ValueError("need 0 < tau_min < tau_max < 1")
        self.register_buffer("bounds", torch.as_tensor(np.asarray(bounds), dtype=torch.float32))
        self.tau_max = float(tau_max)
        self.tau_min = float(tau_min)

    def forward(self, logits: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
        p = logits.sigmoid().clamp(1e-6, 1.0 - 1e-6)
        s = shifts.to(dtype=p.dtype)
        k1, k2, k3, k4 = (self.bounds[:, i].view(1, 1, -1) for i in range(4))
        # Unused regions are clamped so the log arguments stay in (0, 1].
        fade_in = ((k2 - s) / (k2 - k1)).clamp(0, 1)
        fade_out = ((s - k3) / (k4 - k3)).clamp(0, 1)
        # Paper eq. (4): -ln(s/K3 + (K3 - s)/K3 * p). At the spot the score must be high;
        # the requirement relaxes to no loss at all by K3.
        soft = (s / k3).clamp(0, 1)
        far = -torch.log(1 - p)
        before = -torch.log((1 - fade_in * p).clamp(min=1e-6))
        after = -torch.log((soft + (1 - soft) * p).clamp(min=1e-6))
        settle = -torch.log((1 - fade_out * p).clamp(min=1e-6))
        loss = torch.where(
            s <= k1, far,
            torch.where(s <= k2, before,
                        torch.where(s < 0, torch.zeros_like(p),
                                    torch.where(s < k3, after, torch.where(s < k4, settle, far)))))
        on_event = (s >= 0) & (s < k3)
        loss = loss + torch.where(on_event, math.log(self.tau_max), math.log(1.0 - self.tau_min))
        return loss.clamp(min=0).mean()
