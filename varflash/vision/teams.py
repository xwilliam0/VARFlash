"""Split detected people into teams by jersey colour.

The torso region of each box is converted to Lab (perceptually uniform, so
Euclidean distance ~ visual difference), grass-green pixels are masked out and
the median colour is clustered. The two biggest clusters are the teams; the
rest (referees, goalkeepers in odd kits) are labelled "other".
"""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans


def jersey_color(frame: np.ndarray, box) -> np.ndarray | None:
    """Median Lab colour of the torso inside an (x1, y1, x2, y2) box, ignoring grass."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    w, h = x2 - x1, y2 - y1
    crop = frame[max(y1 + int(0.15 * h), 0) : max(y1 + int(0.5 * h), 0), max(x1 + int(0.25 * w), 0) : max(x1 + int(0.75 * w), 0)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    grass = (hsv[:, 0] >= 35) & (hsv[:, 0] <= 85) & (hsv[:, 1] > 60)
    keep = lab[~grass]
    if len(keep) < 10:
        return None
    return np.median(keep, axis=0)


def hex_to_lab(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    rgb = np.array([[[int(h[i : i + 2], 16) for i in (0, 2, 4)]]], np.uint8)
    return cv2.cvtColor(rgb[:, :, ::-1].copy(), cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)


def assign_teams(colors: np.ndarray, attacking_color: str | None = None, seed: int = 0) -> np.ndarray:
    """Label each colour row 'A' (attacking), 'B' (defending) or 'other'.

    With ``attacking_color`` (hex, e.g. "#75AADB" for Argentina) the team cluster
    nearest that colour becomes 'A'; otherwise the larger team cluster is 'A'.
    """
    n = len(colors)
    if n < 2:
        return np.array(["A"] * n)
    k = 3 if n >= 6 else 2
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(colors)
    sizes = np.bincount(km.labels_, minlength=k)
    teams = np.argsort(-sizes)[:2]
    if attacking_color is not None:
        target = hex_to_lab(attacking_color)
        dists = [np.linalg.norm(km.cluster_centers_[c] - target) for c in teams]
        attack, defend = (teams[0], teams[1]) if dists[0] <= dists[1] else (teams[1], teams[0])
    else:
        attack, defend = teams
    labels = np.full(n, "other", dtype=object)
    labels[km.labels_ == attack] = "A"
    labels[km.labels_ == defend] = "B"
    return labels
