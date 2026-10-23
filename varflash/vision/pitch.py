"""FIFA-standard pitch model (metres) and a top-down minimap renderer.

Coordinates: origin at the top-left corner flag as seen from the main camera,
x along the touchline (0..105), y across the pitch (0..68).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

LENGTH, WIDTH = 105.0, 68.0
HALF = LENGTH / 2
BOX_DEPTH, BOX_WIDTH = 16.5, 40.32
SIX_DEPTH, SIX_WIDTH = 5.5, 18.32
CIRCLE_R = 9.15
PENALTY_SPOT = 11.0

_box_y0, _box_y1 = (WIDTH - BOX_WIDTH) / 2, (WIDTH + BOX_WIDTH) / 2
_six_y0, _six_y1 = (WIDTH - SIX_WIDTH) / 2, (WIDTH + SIX_WIDTH) / 2
_arc_dy = math.sqrt(CIRCLE_R**2 - (BOX_DEPTH - PENALTY_SPOT) ** 2)

KEYPOINTS: dict[str, tuple[float, float]] = {
    "corner_tl": (0, 0),
    "corner_bl": (0, WIDTH),
    "corner_tr": (LENGTH, 0),
    "corner_br": (LENGTH, WIDTH),
    "halfway_top": (HALF, 0),
    "halfway_bottom": (HALF, WIDTH),
    "center_spot": (HALF, WIDTH / 2),
    "circle_top": (HALF, WIDTH / 2 - CIRCLE_R),
    "circle_bottom": (HALF, WIDTH / 2 + CIRCLE_R),
    "circle_left": (HALF - CIRCLE_R, WIDTH / 2),
    "circle_right": (HALF + CIRCLE_R, WIDTH / 2),
    "lbox_goal_top": (0, _box_y0),
    "lbox_goal_bottom": (0, _box_y1),
    "lbox_top": (BOX_DEPTH, _box_y0),
    "lbox_bottom": (BOX_DEPTH, _box_y1),
    "lsix_goal_top": (0, _six_y0),
    "lsix_goal_bottom": (0, _six_y1),
    "lsix_top": (SIX_DEPTH, _six_y0),
    "lsix_bottom": (SIX_DEPTH, _six_y1),
    "lpen_spot": (PENALTY_SPOT, WIDTH / 2),
    "larc_top": (BOX_DEPTH, WIDTH / 2 - _arc_dy),
    "larc_bottom": (BOX_DEPTH, WIDTH / 2 + _arc_dy),
    "rbox_goal_top": (LENGTH, _box_y0),
    "rbox_goal_bottom": (LENGTH, _box_y1),
    "rbox_top": (LENGTH - BOX_DEPTH, _box_y0),
    "rbox_bottom": (LENGTH - BOX_DEPTH, _box_y1),
    "rsix_goal_top": (LENGTH, _six_y0),
    "rsix_goal_bottom": (LENGTH, _six_y1),
    "rsix_top": (LENGTH - SIX_DEPTH, _six_y0),
    "rsix_bottom": (LENGTH - SIX_DEPTH, _six_y1),
    "rpen_spot": (LENGTH - PENALTY_SPOT, WIDTH / 2),
    "rarc_top": (LENGTH - BOX_DEPTH, WIDTH / 2 - _arc_dy),
    "rarc_bottom": (LENGTH - BOX_DEPTH, WIDTH / 2 + _arc_dy),
}


def draw_pitch(scale: float = 8.0, margin: float = 4.0) -> tuple[np.ndarray, callable]:
    """Blank minimap image and a function mapping pitch metres -> minimap pixels."""
    w = int((LENGTH + 2 * margin) * scale)
    h = int((WIDTH + 2 * margin) * scale)
    img = np.full((h, w, 3), (40, 120, 40), np.uint8)

    def px(x, y):
        return int(round((x + margin) * scale)), int(round((y + margin) * scale))

    white, t = (240, 240, 240), max(1, int(scale / 4))
    cv2.rectangle(img, px(0, 0), px(LENGTH, WIDTH), white, t)
    cv2.line(img, px(HALF, 0), px(HALF, WIDTH), white, t)
    cv2.circle(img, px(HALF, WIDTH / 2), int(CIRCLE_R * scale), white, t)
    for x0, sign in ((0, 1), (LENGTH, -1)):
        cv2.rectangle(img, px(x0, _box_y0), px(x0 + sign * BOX_DEPTH, _box_y1), white, t)
        cv2.rectangle(img, px(x0, _six_y0), px(x0 + sign * SIX_DEPTH, _six_y1), white, t)
        cv2.circle(img, px(x0 + sign * PENALTY_SPOT, WIDTH / 2), t + 1, white, -1)
    return img, px
