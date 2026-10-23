"""Single-frame AI VAR review: detections + homography -> offside call + overlays."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

from .homography import Homography
from .offside import DEFAULT_POSITION_SIGMA_M, OffsideCall, judge_offside
from .pitch import WIDTH, draw_pitch
from .teams import assign_teams, jersey_color

TEAM_BGR = {"A": (255, 160, 60), "B": (60, 60, 230), "other": (0, 220, 220)}


@dataclass
class Review:
    call: OffsideCall
    players: pd.DataFrame  # detections with team + pitch coordinates
    overlay: np.ndarray
    minimap: np.ndarray


def review_frame(
    frame: np.ndarray,
    detections: pd.DataFrame,
    homography: Homography,
    attack_right: bool,
    attacking_color: str | None = None,
    receiver_xy: tuple[float, float] | None = None,
    keeper_off_screen: bool = True,
    position_sigma_m: float = DEFAULT_POSITION_SIGMA_M,
) -> Review:
    people = detections[detections["kind"] == "person"].copy()
    colors = [jersey_color(frame, r[["x1", "y1", "x2", "y2"]].to_numpy()) for _, r in people.iterrows()]
    keep = [c is not None for c in colors]
    people = people[keep].reset_index(drop=True)
    people["team"] = assign_teams(np.stack([c for c in colors if c is not None]), attacking_color)
    pitch_xy = homography.to_pitch(people[["u", "v"]].to_numpy())
    people["px"], people["py"] = pitch_xy[:, 0], pitch_xy[:, 1]

    ball = detections[detections["kind"] == "ball"].sort_values("conf", ascending=False)
    ball_xy = homography.to_pitch(ball[["u", "v"]].to_numpy()[:1])[0] if len(ball) else None
    ball_x = float(ball_xy[0]) if ball_xy is not None else None

    att = people[people["team"] == "A"].reset_index(drop=True)
    dfn = people[people["team"] == "B"].reset_index(drop=True)
    receiver = None
    if receiver_xy is not None and len(att):
        receiver = int(np.argmin(np.hypot(att["u"] - receiver_xy[0], att["v"] - receiver_xy[1])))
    call = judge_offside(
        att["px"].to_numpy(), dfn["px"].to_numpy(), ball_x, attack_right, receiver,
        keeper_off_screen, position_sigma_m,
    )

    overlay = _draw_overlay(frame, people, att, dfn, call, homography)
    minimap = _draw_minimap(people, call, ball_xy)
    return Review(call, people, overlay, minimap)


def _draw_overlay(frame, people, att, dfn, call: OffsideCall, homography: Homography) -> np.ndarray:
    out = frame.copy()
    for r in people.itertuples():
        cv2.rectangle(out, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), TEAM_BGR[r.team], 2)
    a, b = homography.to_image([[call.line_x, 0], [call.line_x, WIDTH]])
    cv2.line(out, tuple(int(v) for v in a), tuple(int(v) for v in b), (0, 255, 255), 2)
    if call.attacker_index is not None:
        r = att.iloc[call.attacker_index]
        cv2.circle(out, (int(r.u), int(r.v)), 10, (0, 0, 255) if call.offside else (0, 255, 0), 3)
    if call.second_last_index is not None:
        r = dfn.iloc[call.second_last_index]
        cv2.circle(out, (int(r.u), int(r.v)), 10, (255, 255, 255), 2)
    verdict = "OFFSIDE" if call.offside else "ONSIDE"
    text = f"{verdict}  margin {call.margin_m:+.2f} m  P(offside)={call.probability:.2f}"
    cv2.rectangle(out, (10, 10), (20 + 14 * len(text), 55), (0, 0, 0), -1)
    cv2.putText(out, text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return out


def _draw_minimap(people, call: OffsideCall, ball_xy) -> np.ndarray:
    img, px = draw_pitch()
    cv2.line(img, px(call.line_x, 0), px(call.line_x, WIDTH), (0, 255, 255), 2)
    for r in people.itertuples():
        cv2.circle(img, px(r.px, r.py), 6, TEAM_BGR[r.team], -1)
    if ball_xy is not None:
        cv2.circle(img, px(*ball_xy), 4, (255, 255, 255), -1)
    return img
