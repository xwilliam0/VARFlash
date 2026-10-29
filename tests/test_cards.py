import json
import pickle

import cv2
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from varflash.cards import clips, train
from varflash.cards.clips import CLIP_FRAMES, read_windows
from varflash.cards.model import CardNet
from varflash.cards.rescore import _features
from varflash.cards.train import ClipDataset, clip_labels


def test_read_windows_frame_grid(tmp_path):
    p = tmp_path / "idx.mkv"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"FFV1"), 25, (64, 32))
    for i in range(250):
        w.write(np.full((32, 64, 3), i, np.uint8))
    w.release()
    (clip,) = read_windows(p, [5.0], half_s=2.0, fps=4, jpeg_quality=100)
    got = [int(cv2.imdecode(np.frombuffer(b, np.uint8), 1).mean().round()) for b in clip]
    assert got == [int(round((3 + k / 4) * 25)) for k in range(16)]


def test_dataset_targets_follow_event_offset(tmp_path, monkeypatch):
    black = cv2.imencode(".jpg", np.zeros((224, 398, 3), np.uint8))[1].tobytes()
    (tmp_path / "c.pkl").write_bytes(pickle.dumps([black] * CLIP_FRAMES))
    monkeypatch.setattr(clips, "CLIPS_DIR", tmp_path)
    monkeypatch.setattr(train, "load_clip", lambda cid: clips.load_clip(cid, tmp_path))
    idx = pd.DataFrame([{"clip_id": "c", "events": json.dumps([[2.0, 1]])}])
    x, y = ClipDataset(idx, frames=48, train=False, radius=0)[0]
    assert x.shape == (48, 3, 224, 398)
    # Centre is frame 24 of the 48-frame window; +2 s at 4 fps = +8 frames.
    assert y[:, 1].nonzero().flatten().tolist() == [32] and y[:, 0].sum() == 0


def test_clip_labels_window():
    assert clip_labels(json.dumps([[3.0, 0], [9.0, 1]]), 4.0).tolist() == [1.0, 0.0]


def test_cardnet_shapes():
    m = CardNet("resnet18", pretrained=False).eval()
    with torch.no_grad():
        assert m(torch.zeros(1, 4, 3, 64, 112)).shape == (1, 4, 2)


def test_fusion_features_are_logits():
    df = pd.DataFrame({"sp_yellow": [0.5], "sp_red": [0.5], "pv_yellow": [0.5], "pv_red": [0.9]})
    f = _features(df)
    assert f.shape == (1, 4) and abs(f[0, 0]) < 1e-6 and f[0, 3] > 2
