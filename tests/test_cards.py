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
from varflash.cards.train import ClipDataset, clip_labels, live_alert_scores, live_scores


def test_read_windows_frame_grid(tmp_path):
    p = tmp_path / "idx.mkv"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"FFV1"), 25, (64, 32))
    for i in range(250):
        w.write(np.full((32, 64, 3), i, np.uint8))
    w.release()
    (clip,) = read_windows(p, [5.0], half_s=2.0, fps=4, jpeg_quality=100)
    got = [int(cv2.imdecode(np.frombuffer(b, np.uint8), 1).mean().round()) for b in clip]
    assert got == [int(np.floor((3 + k / 4) * 25 + 0.5)) for k in range(16)]  # nearest frame, ties later


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


def test_delay_moves_targets_later(tmp_path, monkeypatch):
    black = cv2.imencode(".jpg", np.zeros((224, 398, 3), np.uint8))[1].tobytes()
    (tmp_path / "c.pkl").write_bytes(pickle.dumps([black] * CLIP_FRAMES))
    monkeypatch.setattr(clips, "CLIPS_DIR", tmp_path)
    monkeypatch.setattr(train, "load_clip", lambda cid: clips.load_clip(cid, tmp_path))
    idx = pd.DataFrame([{"clip_id": "c", "events": json.dumps([[2.0, 1]])}])
    _, y = ClipDataset(idx, frames=48, train=False, radius=0, delay=8)[0]
    assert y[:, 1].nonzero().flatten().tolist() == [40]


def test_live_scores_are_past_only_and_match_a_window():
    torch.manual_seed(0)
    m = CardNet("resnet18", pretrained=False, causal=True).eval()
    x = torch.rand(1, 12, 3, 32, 48) * 255
    y = x.clone()
    y[:, 8:] = torch.rand(1, 4, 3, 32, 48) * 255
    a, b = live_scores(m, x, frames=5), live_scores(m, y, frames=5)
    assert a.shape == (1, 12, 2)
    torch.testing.assert_close(a[:, :8], b[:, :8])
    assert not torch.allclose(a[:, 8:], b[:, 8:])
    with torch.no_grad():
        # Frame 9 sees frames 5..9; frame 3 sees 0..3 (not a full window yet).
        torch.testing.assert_close(a[:, 9], torch.sigmoid(m(x[:, 5:10]))[:, -1])
        torch.testing.assert_close(a[:, 3], torch.sigmoid(m(x[:, :4]))[:, -1])


def test_live_alert_scores_only_use_frames_seen_by_then():
    live = np.zeros((2, CLIP_FRAMES, 2), np.float32)
    live[0, 48 + 8 + 12, 1] = 0.9  # card at +2 s (frame 56); model fires 3 s later
    live[1, 48 + 4, 0] = 0.7        # no card: centre is the reference
    idx = pd.DataFrame({"events": [json.dumps([[2.0, 1]]), "[]"]})
    s = live_alert_scores(live, idx, delays_s=(2.0, 3.0))
    assert s[2.0][0, 1] == 0 and s[3.0][0, 1] == np.float32(0.9)
    assert s[2.0][1, 0] == np.float32(0.7)


def test_tta_on_a_mirror_symmetric_clip_changes_nothing(tmp_path, monkeypatch):
    img = np.zeros((224, 398, 3), np.uint8)
    img[:, :100] = 200
    img[:, -100:] = 200  # left-right symmetric
    jpg = cv2.imencode(".png", img)[1].tobytes()
    (tmp_path / "c.pkl").write_bytes(pickle.dumps([jpg] * CLIP_FRAMES))
    monkeypatch.setattr(train, "load_clip", lambda cid: clips.load_clip(cid, tmp_path))
    idx = pd.DataFrame([{"clip_id": "c", "events": "[]"}])
    torch.manual_seed(0)
    for causal in (False, True):
        m = CardNet("resnet18", pretrained=False, causal=causal).eval()
        a = train.clip_scores(m, idx, frames=8, num_workers=0)
        b = train.clip_scores(m, idx, frames=8, num_workers=0, tta=True)
        assert a.shape == b.shape == (1, 2)
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)


def test_card_probability_platt_identity_and_shrink():
    from sklearn.linear_model import LogisticRegression

    from varflash.cards.rescore import card_probability

    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.uniform(0.01, 0.99, (200, 4)), columns=["sp_yellow", "sp_red", "pv_yellow", "pv_red"])
    y = (df["pv_yellow"] > 0.5).astype(int).to_numpy()
    fusion = [LogisticRegression().fit(np.log(df.to_numpy() / (1 - df.to_numpy())), y)] * 2
    raw = card_probability(df, fusion)
    np.testing.assert_allclose(card_probability(df, fusion, np.array([[1.0, 0.0]] * 2)), raw, rtol=1e-3, atol=1e-4)
    shrunk = card_probability(df, fusion, np.array([[0.5, 0.0]] * 2))  # slope < 1 pulls towards 0.5
    assert (np.abs(shrunk - 0.5) <= np.abs(raw - 0.5) + 1e-9).all()
    assert (np.argsort(shrunk[:, 0]) == np.argsort(raw[:, 0])).all()  # monotone: AP unchanged


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
