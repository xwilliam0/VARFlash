import numpy as np
import pandas as pd
import torch

from varflash.cards.model import CardNet
from varflash.goals.clips import CLIP_FRAMES, PRE_S, shot_episodes
from varflash.goals.stream import goal_alerts, to_grid
from varflash.goals.train import delay_report, frame_targets, live_scores


def test_shot_episodes_merge_and_goal_assignment():
    ev = [(100.0, "Shots on target"), (102.0, "Shots on target"), (102.6, "Goal"),
          (300.0, "Shots off target"), (500.0, "Goal")]
    eps = shot_episodes(ev)
    assert [round(e["anchor_s"], 1) for e in eps] == [100.0, 300.0, 499.4]
    assert [e["kind"] for e in eps] == ["goal", "shot_off", "goal"]
    assert eps[0]["goal_s"] == 102.6 and np.isnan(eps[1]["goal_s"])


def test_frame_targets_step_at_goal():
    y = frame_targets(goal_s=11.0, start_s=10.0, n=30)
    assert y[:12].sum() == 0 and y[13:].all()  # 12.5 fps: frame 12.5 is the crossing
    assert frame_targets(float("nan"), 0.0, 5).sum() == 0


def test_live_scores_match_causal_forward():
    torch.manual_seed(0)
    m = CardNet("resnet18", n_classes=1, pretrained=False, causal=True).eval()
    x = torch.rand(1, 12, 3, 32, 48) * 255
    live = live_scores(m, x, frames=5)
    assert torch.isnan(live[0, :4]).all()
    with torch.no_grad():
        ref = torch.sigmoid(m(x[:, 7:12])[0, -1, 0])
    assert torch.allclose(live[0, 11], ref, atol=1e-5)


def test_to_grid_is_causal_max():
    t = np.arange(0, 3, 1 / 12.5)
    p = np.where(np.isclose(t, 1.04), 0.9, 0.1)
    g = to_grid(t, p, fps=2, n=6)
    assert g[3] == np.float32(0.9) and g[2] == np.float32(0.1)  # 1.04 s is first known at grid time 1.5 s


def test_goal_alerts_merge_replays_and_fire_on_crossing():
    s = np.zeros(400, np.float32)
    s[100:104] = [0.2, 0.6, 0.95, 0.9]  # goal: crosses 0.9 at 51 s (2 fps)
    s[160:162] = 0.97  # replay 30 s later: same alert
    s[250] = 0.8  # below threshold: no alert
    s[300] = 0.92  # separate alert
    a = goal_alerts(s, fps=2)
    assert len(a) == 2
    assert a.loc[0, "fire_s"] == 51.0 and a.loc[0, "time_s"] == 50.0 and a.loc[0, "confidence"] == np.float32(0.97)
    assert a.loc[1, "fire_s"] == 150.0


def test_delay_report_perfect_detector():
    idx = pd.DataFrame({"anchor_s": [100.0, 200.0], "start_s": [100 - PRE_S, 200 - PRE_S],
                        "goal_s": [100.6, np.nan], "kind": ["goal", "shot_on"]})
    s = np.zeros((2, CLIP_FRAMES))
    s[0, int((PRE_S + 1.0) * 12.5):] = 0.9  # fires 0.4 s after the ball crosses
    rep, _ = delay_report(s, idx, gap_s=0.6)
    rep = rep.set_index("delay_s")
    assert rep.loc[1.0, "recall@p0.9"] == 1.0 and rep.loc[0.0, "recall@p0.9"] == 0.0
