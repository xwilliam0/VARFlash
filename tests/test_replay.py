import pandas as pd

from varflash.backtest.clock import MatchClock, SegmentClock
from varflash.replay.build import attach_teams, scorecard


def test_utc_to_video_inverts_video_to_utc_and_skips_halftime():
    clock = MatchClock("2026-07-12T01:00:00Z", "2026-07-12T02:05:00Z", video_kickoff_s=30, video_second_half_s=3600)
    for t in (30.0, 1500.0, 3599.0, 3600.0, 6000.0):
        assert abs(clock.utc_to_video(clock.video_to_utc(t)) - t) < 1e-6
    # 01:59:30 UTC is in the real half-time break, but the video has already reached the 2nd half there.
    assert clock.utc_to_video(pd.Timestamp("2026-07-12T01:59:30Z")) is None


def test_segment_clock_maps_each_clip_around_its_anchor():
    base = MatchClock("2026-07-12T01:00:00Z")
    segs = [
        {"start_s": 0, "end_s": 100, "anchor_video_s": 50, "anchor_utc": "2026-07-12T01:10:00Z", "minute": 10},
        {"start_s": 100, "end_s": 200, "anchor_video_s": 150, "anchor_utc": "2026-07-12T03:32:00Z", "minute": 112},
    ]
    clock = SegmentClock(base, segs)
    assert clock.video_to_utc(60) == pd.Timestamp("2026-07-12T01:10:10Z")
    assert clock.video_to_utc(140) == pd.Timestamp("2026-07-12T03:31:50Z")
    assert clock.utc_to_video(pd.Timestamp("2026-07-12T03:32:30Z")) == 180
    assert clock.utc_to_video(pd.Timestamp("2026-07-12T02:00:00Z")) is None  # between clips: no footage
    assert abs(clock.match_minute(pd.Timestamp("2026-07-12T03:32:30Z")) - 112.5) < 1e-9


def test_attach_teams_uses_nearest_same_type_reference():
    alerts = pd.DataFrame({"label": ["Goal", "Goal", "Sending off"], "time_s": [100.0, 900.0, 105.0]})
    truth = pd.DataFrame({"video_s": [95.0, 160.0, 400.0], "label": ["Goal", "Goal", "Sending off"],
                          "team": ["home", "away", "away"], "visible": True})
    out = attach_teams(alerts, truth, tol_s=60)
    assert out["team"].tolist() == ["home", None, None]
    assert out.loc[0, "ref_s"] == 95.0


def test_scorecard_counts_catches_false_alarms_and_latency():
    alerts = pd.DataFrame({"label": ["Goal", "Goal", "Goal"], "time_s": [100.0, 500.0, 2000.0],
                           "fire_s": [110.0, 510.0, 2010.0], "confidence": [0.9, 0.8, 0.1]})
    truth = pd.DataFrame({"video_s": [98.0, 1200.0], "label": ["Goal", "Goal"], "team": ["home", "away"], "visible": True})
    (row,) = scorecard(alerts, truth, min_conf=0.3, tol_s=60)
    assert (row["events"], row["caught"], row["alerts"], row["false_alarms"]) == (2, 1, 2, 1)
    assert row["median_latency_s"] == 12.0
