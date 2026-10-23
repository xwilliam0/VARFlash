"""Offside geometry on the 2D pitch plane (Law 11, simplified).

An attacker is in an offside position at the moment a team-mate plays the
ball if they are in the opponents' half and nearer the opponents' goal line
than both the ball and the second-last opponent. Positions come from player
foot points, so each carries ~0.2-0.5 m of error from detection and the
homography. Instead of a hard yes/no we report P(offside) under Gaussian
position noise, which is what a trading strategy actually needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .pitch import HALF, LENGTH

DEFAULT_POSITION_SIGMA_M = 0.3


@dataclass
class OffsideCall:
    offside: bool
    probability: float
    margin_m: float  # attacker depth beyond the offside line (positive = offside)
    line_x: float  # offside line in pitch x (metres)
    attacker_index: int | None
    second_last_index: int | None


def _depth(x: np.ndarray, attack_right: bool) -> np.ndarray:
    """Distance travelled towards the defended goal line (bigger = closer to goal)."""
    return x if attack_right else LENGTH - x


def offside_line_depth(
    defender_x: np.ndarray, ball_x: float | None, attack_right: bool, keeper_off_screen: bool = True
) -> tuple[float, int | None]:
    """Depth of the offside line and index of the defender who sets it.

    ``keeper_off_screen``: broadcast VAR frames usually don't show the
    goalkeeper; assume they're the last defender, so the deepest *visible*
    defender is the second-last opponent.
    """
    d = _depth(np.asarray(defender_x, float), attack_right)
    order = np.argsort(-d)
    rank = 0 if keeper_off_screen else 1
    if len(d) > rank:
        idx = int(order[rank])
        line, who = d[idx], idx
    else:
        line, who = 0.0, None
    if line < HALF:  # can't be offside in your own half
        line, who = HALF, None
    if ball_x is not None:
        ball_d = float(_depth(np.array([ball_x]), attack_right)[0])
        if ball_d > line:
            line, who = ball_d, None
    return float(line), who


def judge_offside(
    attacker_x,
    defender_x,
    ball_x: float | None = None,
    attack_right: bool = True,
    receiver_index: int | None = None,
    keeper_off_screen: bool = True,
    position_sigma_m: float = DEFAULT_POSITION_SIGMA_M,
) -> OffsideCall:
    """Judge the receiving attacker (default: the most advanced one)."""
    att = _depth(np.asarray(attacker_x, float), attack_right)
    if len(att) == 0:
        raise ValueError("no attackers")
    line, who = offside_line_depth(defender_x, ball_x, attack_right, keeper_off_screen)
    i = int(np.argmax(att)) if receiver_index is None else receiver_index
    margin = float(att[i] - line)
    # Margin is a difference of two noisy positions.
    sigma = np.sqrt(2) * position_sigma_m
    prob = float(norm.cdf(margin / sigma))
    line_x = line if attack_right else LENGTH - line
    return OffsideCall(margin > 0, prob, margin, float(line_x), i, who)
