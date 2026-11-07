import json

import cv2
import numpy as np
import pandas as pd

from varflash.offside import train as ot
from varflash.offside.clips import CLIP_FRAMES, PRE_S, offsides_near


def test_offsides_near_keeps_offsets_inside_the_clip():
    labels = [(100.0, "Offside"), (95.0, "Foul"), (120.0, "Offside"), (80.0, "Offside")]
    assert json.loads(offsides_near(labels, 100.0)) == [0.0]
    assert json.loads(offsides_near(labels, 108.0)) == [-8.0, 12.0]
    assert json.loads(offsides_near(labels, 91.0)) == [9.0]  # 120 s is 29 s after: outside the clip


def test_clip_targets_sit_on_the_offside_frame(monkeypatch):
    black = cv2.imencode(".jpg", np.zeros((224, 398, 3), np.uint8))[1].tobytes()
    monkeypatch.setattr(ot, "load_clip", lambda cid: [black] * CLIP_FRAMES)
    cfg = ot.OffsideTrainConfig(frames=48, lead=16, radius=0)
    index = pd.DataFrame({"clip_id": ["a", "b"], "offsides": [json.dumps([0.0]), json.dumps([2.5])]})
    ds = ot.OffsideClips(index, cfg, train=False)
    x, y = ds[0]
    assert x.shape == (48, 3, 224, 398)
    assert y[:, 0].nonzero().flatten().tolist() == [16]  # the candidate frame
    _, y = ds[1]
    assert y[:, 0].nonzero().flatten().tolist() == [16 + 10]
    assert int(PRE_S * 4) - cfg.lead >= 0
