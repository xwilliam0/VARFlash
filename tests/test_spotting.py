import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from varflash.spotting.context_loss import ContextAwareLoss, bounds_for, time_shifts
from varflash.spotting.data import CLASS_SETS, Half, WindowDataset, frame_targets
from varflash.spotting.features import KerasResNet152, _keras_layer_map, preprocess
from varflash.spotting.infer import SpotterEnsemble, event_scores, frame_scores, load_spotter, peak_pick
from varflash.spotting.kickoffs import propose_kickoffs
from varflash.spotting.metrics import calibration_report, first_crossings, latency_report, match_predictions
from varflash.spotting.model import TemporalSpotter
from varflash.spotting.train import EMA, augment


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


def test_centred_frame_scores_use_only_the_window_centre():
    torch.manual_seed(0)
    m = TemporalSpotter(in_dim=8, dim=16, n_classes=2).eval()
    x = np.random.default_rng(0).normal(size=(203, 8)).astype(np.float32)
    s = frame_scores(m, x, window=40, edge=10)
    assert s.shape == (203, 2)
    # Frame 50 is scored by the window over padded frames [40, 80), i.e. original frames [30, 70).
    with torch.no_grad():
        ref = torch.sigmoid(m(torch.from_numpy(x[30:70])[None]))[0, 20].numpy()
    np.testing.assert_allclose(s[50], ref, atol=1e-5)
    # The first frames see zero padding before the half, as in training.
    pad = np.concatenate([np.zeros((10, 8), np.float32), x[:30]])
    with torch.no_grad():
        ref0 = torch.sigmoid(m(torch.from_numpy(pad)[None]))[0, 10].numpy()
    np.testing.assert_allclose(s[0], ref0, atol=1e-5)


def test_padded_windows_reach_past_the_half():
    feats = np.ones((100, 4), np.float32)
    h = Half("g", 1, feats, frame_targets(100, [(0, 0), (99, 0)], radius=0, n_classes=1), [(0, 0), (99, 0)])
    ds = WindowDataset([h], window=40, stride=20, train=False, pad=10)
    starts = [s for _, s in ds.index]
    assert starts[0] == -10 and starts[-1] == 70
    x, y = ds[0]
    assert x.shape == (40, 4) and x[:10].sum() == 0 and x[10:].sum() == 30 * 4
    assert y[10, 0] == 1  # the kick-off frame sits at the first supervised position
    x, y = ds[len(ds) - 1]
    assert y[29, 0] == 1 and x[30:].sum() == 0  # the last frame is the last supervised position


def test_delay_shifts_targets_and_scores_back():
    y = frame_targets(20, [(5, 0)], radius=0, n_classes=1, delay=3)
    assert y[8, 0] == 1 and y.sum() == 1
    np.testing.assert_array_equal(event_scores(y, 3)[5], [1.0])


def test_augment_masks_spans_and_mixes_targets():
    torch.manual_seed(0)
    np.random.seed(0)
    x, y = torch.ones(8, 40, 4), torch.zeros(8, 40, 3)
    y[0] = 1.0
    xm, ym = augment(x, y, time_mask=5)
    zero = (xm == 0).all(-1)
    assert torch.equal(ym, y) and zero.sum(1).max() <= 10 and zero.any()
    assert ((xm == 0) | (xm == 1)).all()
    xa, ya = augment(x, y, mixup_alpha=0.4)
    assert torch.allclose(xa, x)  # all-ones windows stay all-ones under any blend
    assert ya.min() >= 0 and ya.max() <= 1 and torch.allclose(ya.sum(), y.sum())
    xo, yo = augment(x, y)
    assert xo is x and yo is y


def test_gaussian_targets_peak_at_the_delayed_event():
    y = frame_targets(40, [(10, 0), (13, 0)], radius=2, n_classes=1, delay=2, sigma=2.0)[:, 0]
    assert y[12] == 1 and y[15] == 1 and y.max() == 1
    assert np.isclose(y[10], np.exp(-0.5))  # 1 sigma before the first event
    assert y[:6].sum() == 0 and y[22:].sum() == 0  # cut at 3 sigma
    assert np.all(np.diff(y[6:13]) > 0) and np.all(np.diff(y[15:22]) < 0)


def test_ema_tracks_weights_and_copies_int_buffers():
    m = TemporalSpotter(in_dim=4, dim=8, n_classes=1)
    ema = EMA(m, decay=0.9)
    with torch.no_grad():
        for p in m.parameters():
            p.fill_(1.0)
    m.tcn[0].norm.num_batches_tracked.fill_(7)
    for _ in range(200):
        ema.update(m)
    assert all(torch.allclose(p, torch.ones_like(p), atol=1e-6) for p in ema.module.parameters())
    assert ema.module.tcn[0].norm.num_batches_tracked.item() == 7
    assert not any(p.requires_grad for p in ema.module.parameters())


def test_ensemble_averages_member_probabilities():
    torch.manual_seed(0)
    a, b = (TemporalSpotter(in_dim=8, dim=16, n_classes=2).eval() for _ in range(2))
    x = np.random.default_rng(0).normal(size=(300, 8)).astype(np.float32)
    np.testing.assert_allclose(frame_scores(SpotterEnsemble([a, a]).eval(), x, 40), frame_scores(a, x, 40), atol=1e-5)
    mean = (frame_scores(a, x, 40) + frame_scores(b, x, 40)) / 2
    np.testing.assert_allclose(frame_scores(SpotterEnsemble([a, b]).eval(), x, 40), mean, atol=1e-5)
    with pytest.raises(ValueError):
        SpotterEnsemble([a, TemporalSpotter(in_dim=8, dim=16, n_classes=2, causal=True)])


def test_context_shifts_follow_the_dominant_event():
    # Card slice from Cioppa et al.: far before -40, ignore (-20, 0), spot through +20, far after +40.
    bounds = bounds_for(["Yellow card"])
    s = time_shifts(160, [(100, 0)], bounds)[:, 0]
    assert s[100] == 0 and s[90] == -10 and s[75] == -25 and s[50] == -50
    assert s[110] == 10 and s[125] == 25 and s[150] == 50
    # No labels at all: every frame is "far before" (K1), same loss as a distant background.
    assert np.all(time_shifts(5, [], bounds) == bounds[0, 0])
    # Two events. A tie in the two transition zones stays with the action that already happened.
    both = time_shifts(80, [(0, 0), (50, 0)], bounds)[:, 0]
    assert both[25] == 25 and both[26] == 26 - 50
    # Causal delay moves the spot the same way the binary targets do.
    assert time_shifts(20, [(5, 0)], bounds, delay=3)[8, 0] == 0


def test_context_loss_ignores_the_buildup_and_easy_frames():
    bounds = bounds_for(["Yellow card"])
    loss_fn = ContextAwareLoss(bounds)

    def one(shift, p):
        logit = torch.tensor([[[float(np.log(p / (1 - p)))]]], requires_grad=True)
        return logit, loss_fn(logit, torch.tensor([[[shift]]]))

    assert one(0, 0.95)[1].item() == 0.0        # already above tau_max on the spot
    assert one(0, 0.5)[1].item() > 0.0
    assert one(-5, 0.99)[1].item() == 0.0       # just before: the attack might come to nothing
    assert one(-100, 0.05)[1].item() == 0.0     # far away and already quiet
    assert one(-100, 0.5)[1].item() > 0.0
    # Transition before, s=-30, p=0.5: -ln(1 - 0.5*0.5) + ln(0.9) = ln(0.9/0.75).
    torch.testing.assert_close(one(-30, 0.5)[1], torch.tensor(float(np.log(0.9 / 0.75))))
    # Just after (K3 = 20), eq. (4): s=10, p=0.2 -> -ln(0.5 + 0.5*0.2) + ln(0.9).
    torch.testing.assert_close(one(10, 0.2)[1], torch.tensor(float(np.log(0.9 / 0.6))))
    assert one(19, 0.99)[1].item() == 0.0       # end of the zone: still firing is not penalised
    # Transition after (K3=20, K4=40), s=30, p=0.5: -ln(1 - 0.5*0.5) + ln(0.9).
    torch.testing.assert_close(one(30, 0.5)[1], torch.tensor(float(np.log(0.9 / 0.75))))
    logit, value = one(0, 0.2)
    value.backward()
    assert logit.grad is not None and logit.grad.item() != 0.0


def test_context_windows_carry_integer_shifts():
    bounds = bounds_for(["Goal"])
    events = [(10, 0)]
    half = Half("g", 1, np.zeros((30, 4), np.float32), frame_targets(30, events, 1, 1), events,
                shifts=time_shifts(30, events, bounds))
    _, y = WindowDataset([half], window=8, train=False)[2]
    assert y.dtype == torch.int32 and y.shape == (8, 1) and int(y[2, 0]) == 0


def test_load_spotter_opens_ensemble_checkpoints(tmp_path):
    torch.manual_seed(0)
    args = {"in_dim": 8, "dim": 16, "n_classes": 2}
    a, b = (TemporalSpotter(**args).eval() for _ in range(2))
    path = tmp_path / "ens.pt"
    torch.save({"members": [{"model_args": args, "state_dict": m.state_dict()} for m in (a, b)],
                "classes": ["A", "B"], "delay": 0, "config": {"window": 40}}, path)
    model, ckpt = load_spotter(path)
    assert isinstance(model, SpotterEnsemble) and ckpt["classes"] == ["A", "B"]
    x = np.random.default_rng(0).normal(size=(200, 8)).astype(np.float32)
    np.testing.assert_allclose(frame_scores(model, x, 40), frame_scores(SpotterEnsemble([a, b]).eval(), x, 40),
                               atol=1e-6)


def _toy_spotting(seed: int = 0):
    rng = np.random.default_rng(seed)
    games = [f"g{i}" for i in range(6)]
    gts, preds = [], []
    for g in games:
        for half in (1, 2):
            v = f"{g}#{half}"
            for t in rng.uniform(0, 2000, 3):
                gts.append({"video": v, "time_s": t, "label": "Goal"})
                preds.append({"video": v, "time_s": t + rng.normal(0, 8), "label": "Goal",
                              "confidence": rng.uniform(0.3, 1)})
            for t in rng.uniform(0, 2000, 4):
                preds.append({"video": v, "time_s": t, "label": "Goal", "confidence": rng.uniform(0, 0.8)})
    return games, pd.DataFrame(preds), pd.DataFrame(gts)


def test_weighted_ap_matches_plain_ap_and_dropping_a_match():
    from varflash.spotting.bootstrap import prepare, weighted_ap
    from varflash.spotting.metrics import average_precision

    games, preds, gts = _toy_spotting()
    prep = prepare(preds, gts, ["Goal"], games, tolerances=(10,))
    assert np.isclose(weighted_ap(*prep[("Goal", 10)], np.ones(len(games))), average_precision(preds, gts, 10))
    w = np.ones(len(games))
    w[2] = 0
    keep = lambda df: df[~df["video"].str.startswith("g2#")]
    assert np.isclose(weighted_ap(*prep[("Goal", 10)], w), average_precision(keep(preds), keep(gts), 10))


def test_bootstrap_compare_identical_models_differ_by_zero():
    from varflash.spotting.bootstrap import compare

    games, preds, gts = _toy_spotting()
    aps, diffs = compare({"a": [preds], "b": [preds.copy()]}, gts, ["Goal"], games, baseline="a", n_boot=50)
    row = aps[(aps.model == "a") & (aps["class"] == "Goal")].iloc[0]
    assert row.ci_low <= row.value <= row.ci_high
    assert (diffs[["value", "ci_low", "ci_high"]].abs() < 1e-12).all().all()
    assert (diffs["p_not_better"] == 1.0).all()


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
