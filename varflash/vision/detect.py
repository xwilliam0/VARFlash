"""Player/ball detection and multi-object tracking with Ultralytics YOLO + ByteTrack.

Works with the stock COCO weights (classes ``person`` and ``sports ball``) out
of the box, or with a detector fine-tuned on football footage whose classes
are e.g. player / goalkeeper / referee / ball.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COCO_PERSON, COCO_BALL = 0, 32
PERSON_NAMES = {"person", "player", "goalkeeper", "referee"}
BALL_NAMES = {"sports ball", "ball"}


def _load(weights: str):
    from ultralytics import YOLO

    return YOLO(weights)


def _rows(result, frame_idx: int, fps: float | None, tracked: bool) -> list[dict]:
    names = result.names
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    ids = boxes.id.cpu().numpy().astype(int) if tracked and boxes.id is not None else np.full(len(cls), -1)
    rows = []
    for (x1, y1, x2, y2), c, p, tid in zip(xyxy, cls, conf, ids):
        name = names[c]
        kind = "ball" if name in BALL_NAMES else "person" if name in PERSON_NAMES else None
        if kind is None:
            continue
        rows.append(
            {
                "frame": frame_idx,
                "t": frame_idx / fps if fps else np.nan,
                "track_id": tid,
                "kind": kind,
                "label": name,
                "conf": float(p),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                # Feet touch the pitch plane; ball uses its centre.
                "u": (x1 + x2) / 2,
                "v": y2 if kind == "person" else (y1 + y2) / 2,
            }
        )
    return rows


def detect_image(image, weights: str = "yolo11m.pt", conf: float = 0.2, imgsz: int = 1280) -> pd.DataFrame:
    model = _load(weights)
    result = model.predict(image, conf=conf, imgsz=imgsz, verbose=False)[0]
    return pd.DataFrame(_rows(result, 0, None, tracked=False))


def track_video(
    video: Path | str,
    weights: str = "yolo11m.pt",
    conf: float = 0.2,
    imgsz: int = 1280,
    tracker: str = "bytetrack.yaml",
    vid_stride: int = 1,
    max_frames: int | None = None,
) -> pd.DataFrame:
    import cv2

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    model = _load(weights)
    rows: list[dict] = []
    stream = model.track(
        source=str(video), stream=True, conf=conf, imgsz=imgsz, tracker=tracker,
        vid_stride=vid_stride, persist=True, verbose=False,
    )
    for i, result in enumerate(stream):
        rows.extend(_rows(result, i * vid_stride, fps, tracked=True))
        if max_frames and i + 1 >= max_frames:
            break
    return pd.DataFrame(rows)
