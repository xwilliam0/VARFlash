"""Image <-> pitch-plane homography from labelled pitch keypoints.

A broadcast frame is a perspective view of the (planar) pitch, so four or more
pixel/pitch correspondences pin down a 3x3 homography H. Player foot points
mapped through H land in metres on the 2D pitch, where offside is just a
comparison of x coordinates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .pitch import KEYPOINTS


@dataclass
class Homography:
    H: np.ndarray  # image -> pitch
    reprojection_error_m: float
    n_points: int

    @property
    def H_inv(self) -> np.ndarray:
        return np.linalg.inv(self.H)

    def to_pitch(self, pts) -> np.ndarray:
        pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_image(self, pts) -> np.ndarray:
        pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)


def fit_homography(image_pts, pitch_pts) -> Homography:
    image_pts = np.asarray(image_pts, np.float64)
    pitch_pts = np.asarray(pitch_pts, np.float64)
    if len(image_pts) < 4:
        raise ValueError("need at least 4 non-collinear keypoints")
    method = cv2.RANSAC if len(image_pts) > 4 else 0
    H, _ = cv2.findHomography(image_pts, pitch_pts, method, ransacReprojThreshold=1.0)
    if H is None:
        raise ValueError("homography fit failed (points collinear?)")
    proj = cv2.perspectiveTransform(image_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = float(np.sqrt(((proj - pitch_pts) ** 2).sum(axis=1)).mean())
    return Homography(H, err, len(image_pts))


def from_keypoints(named: dict[str, tuple[float, float]]) -> Homography:
    unknown = set(named) - set(KEYPOINTS)
    if unknown:
        raise KeyError(f"unknown keypoints: {sorted(unknown)}")
    names = list(named)
    return fit_homography([named[n] for n in names], [KEYPOINTS[n] for n in names])


def save_calibration(path: Path | str, named: dict[str, tuple[float, float]]) -> None:
    Path(path).write_text(json.dumps({"keypoints": named}, indent=2))


def load_calibration(path: Path | str) -> Homography:
    return from_keypoints({k: tuple(v) for k, v in json.loads(Path(path).read_text())["keypoints"].items()})


def calibrate_interactive(frame: np.ndarray, out_path: Path | str) -> Homography:
    """Click pitch keypoints on a frame. Keys: n=skip keypoint, u=undo, q=finish."""
    names = list(KEYPOINTS)
    picked: dict[str, tuple[float, float]] = {}
    state = {"i": 0}
    win = "calibrate: click the named keypoint (n=skip, u=undo, q=done)"

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and state["i"] < len(names):
            picked[names[state["i"]]] = (float(x), float(y))
            state["i"] += 1

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_click)
    while True:
        vis = frame.copy()
        for n, (x, y) in picked.items():
            cv2.circle(vis, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(vis, n, (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        label = names[state["i"]] if state["i"] < len(names) else "(all done) press q"
        cv2.putText(vis, f"next: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow(win, vis)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("n"):
            state["i"] = min(state["i"] + 1, len(names))
        elif key == ord("u") and picked:
            last = list(picked)[-1]
            picked.pop(last)
            state["i"] = names.index(last)
        elif key == ord("q"):
            break
    cv2.destroyWindow(win)
    save_calibration(out_path, picked)
    return from_keypoints(picked)
