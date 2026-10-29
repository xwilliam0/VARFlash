import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from varflash.spotting.data import CLASS_SETS, frame_targets
from varflash.spotting.features import KerasResNet152, _keras_layer_map, preprocess
from varflash.spotting.infer import event_scores, frame_scores, peak_pick
from varflash.spotting.kickoffs import propose_kickoffs
from varflash.spotting.metrics import calibration_report, first_crossings, latency_report, match_predictions
from varflash.spotting.model import TemporalSpotter


def test_keras_resnet152_shape_and_layers():
    m = KerasResNet152().eval()
    # Keras ResNet152(include_top=False) has 58,370,944 parameters (incl. BN statistics).
    n = sum(p.numel() for p in m.parameters()) + sum(b.numel() for n_, b in m.named_buffers() if "running" in n_)
    assert n == 58_370_944
    layers = _keras_layer_map(m)
    assert len(layers) == 2 + 50 * 6 + 4 * 2  # stem + 50 blocks x (3 conv + 3 bn) + 4 shortcuts x (conv + bn)
    with torch.no_grad():
        assert m(torch.zeros(2, 3, 224, 224)).shape == (2, 2048)


def test_preprocess_reproduces_soccernet_channel_quirk():
    bgr = np.zeros((1, 224, 224, 3), np.uint8)
    bgr[..., 0] = 200  # blue in OpenCV order
    x = preprocess(bgr)  # SoccerNet: BGR fed to an RGB->BGR flip
    assert x.shape == (1, 3, 224, 224)
    assert torch.allclose(x[0, 2, 0, 0], torch.tensor(200 - 123.68))
    assert torch.allclose(preprocess(bgr, swap_rb=False)[0, 0, 0, 0], torch.tensor(200 - 103.939))


def test_causal_model_ignores_the_future():
    torch.manual_seed(0)
    m = TemporalSpotter(in_dim=8, dim=16, n_classes=2, causal=True).eval()
    x = torch.randn(1, 50, 8)
    y = x.clone()
    y[:, 30:] = torch.randn(1, 20, 8)
    with torch.no_grad():
        a, b = m(x), m(y)
    assert torch.allclose(a[:, :30], b[:, :30], atol=1e-6)
    assert not torch.allclose(a[:, 30:], b[:, 30:])


def test_causal_frame_scores_are_live_valid():
    torch.manual_seed(0)
    m = TemporalSpotter(in_dim=8, dim=16, n_classes=2, causal=True).eval()
    x = np.random.default_rng(0).normal(size=(500, 8)).astype(np.float32)
    y = x.copy()
    y[300:] = 0
    a, b = frame_scores(m, x, window=40), frame_scores(m, y, window=40)
    np.testing.assert_allclose(a[:300], b[:300], atol=1e-6)


def test_delay_shifts_targets_and_scores_back():
    y = frame_targets(20, [(5, 0)], radius=0, n_classes=1, delay=3)
    assert y[8, 0] == 1 and y.sum() == 1
    np.testing.assert_array_equal(event_scores(y, 3)[5], [1.0])


def test_cards_class_set_merges_sending_off():
    assert CLASS_SETS["cards"]["Sending off"] == ["Red card", "Yellow->red card"]


def test_peak_pick_nms():
    s = np.zeros((100, 1), np.float32)
    s[[10, 12, 60], 0] = [0.9, 0.5, 0.7]
    p = peak_pick(s, ["Goal"], nms_window_s=5)
    assert p["frame"].tolist() == [10, 60]


def test_match_predictions_one_to_one():
    preds = pd.DataFrame({"video": ["a", "a", "a"], "time_s": [10.0, 11.0, 50.0], "confidence": [0.9, 0.8, 0.7]})
    gts = pd.DataFrame({"video": ["a"], "time_s": [10.5]})
    assert match_predictions(preds, gts, 5).tolist() == [True, False, False]


def test_latency_report():
    scores = np.zeros(400, np.float32)
    scores[130:140] = 0.9  # fires 5 s after the event at 60 s
    scores[300:305] = 0.9  # false alarm
    r = latency_report([("a", scores, np.array([60.0]))], threshold=0.5)
    assert r["alerts"] == 2 and r["precision"] == 0.5
    assert r["median_delay_s"] == 5.0 and r["recall@15s"] == 1.0
    assert first_crossings(scores, 0.5).tolist() == [130, 300]


def test_calibration_report_perfect():
    r = calibration_report(np.array([0.0, 1.0, 1.0]), np.array([0, 1, 1]))
    assert r["brier"] == 0 and r["ece"] == 0


def test_propose_kickoffs_prefers_quiet_restart():
    ev = pd.DataFrame([
        {"time_s": 30.0, "label": "Kick-off", "confidence": 0.9},
        {"time_s": 900.0, "label": "Kick-off", "confidence": 0.8},   # after a goal
        *[{"time_s": 3000.0 + 10 * i, "label": "Throw-in", "confidence": 0.8} for i in range(20)],
        {"time_s": 3300.0, "label": "Kick-off", "confidence": 0.9},  # busy play: likely a goal restart
        {"time_s": 4200.0, "label": "Kick-off", "confidence": 0.85},  # after a quiet half-time
    ])
    p = propose_kickoffs(ev)
    assert p["first_half"]["time_s"].iloc[0] == 30.0
    assert p["second_half"]["time_s"].iloc[0] == 4200.0
