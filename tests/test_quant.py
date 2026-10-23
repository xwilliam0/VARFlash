import pytest

from varflash.backtest.clock import MatchClock, parse_game_time
from varflash.backtest.fees import kalshi_fee
from varflash.quant.poisson import InPlayModel, MatchState, fit_rates, outcome_probs


def test_outcome_probs_sum_to_one_and_symmetry():
    p = outcome_probs(1.3, 1.3)
    assert p.sum() == pytest.approx(1.0)
    assert p[0] == pytest.approx(p[2])


def test_fit_rates_roundtrip():
    target = outcome_probs(1.7, 0.8)
    lh, la = fit_rates(*target)
    assert (lh, la) == pytest.approx((1.7, 0.8), rel=1e-3)


def test_inplay_events_move_probabilities_the_right_way():
    m = InPlayModel.from_market(0.56, 0.29, 0.15)
    s = MatchState(minute=70, home_goals=1, away_goals=1)
    base = m.probs_dict(s)
    after_red = m.probs_dict(s.apply("red_card", "away"))
    assert after_red["home"] > base["home"] and after_red["away"] < base["away"]

    after_goal = m.probs_dict(s.apply("goal", "home"))
    assert after_goal["home"] > 0.7

    disallowed = s.apply("goal", "home").apply("goal_disallowed", "home")
    assert m.probs_dict(disallowed) == pytest.approx(base)


def test_draw_converges_at_full_time():
    m = InPlayModel(1.5, 1.0)
    assert m.probs_dict(MatchState(minute=200, home_goals=2, away_goals=2))["draw"] == pytest.approx(1.0)


def test_kalshi_fee_rounds_up_to_cent():
    assert kalshi_fee(100, 0.50) == pytest.approx(1.75)
    assert kalshi_fee(1, 0.50) == pytest.approx(0.02)  # 0.0175 -> 0.02
    assert kalshi_fee(0, 0.5) == 0.0


def test_match_clock_mapping():
    clock = MatchClock("2026-07-12T01:00:00Z", "2026-07-12T02:05:00Z", video_kickoff_s=30, video_second_half_s=3600)
    assert clock.video_to_utc(30 + 600).strftime("%H:%M") == "01:10"
    assert clock.video_to_utc(3600 + 120).strftime("%H:%M") == "02:07"
    assert clock.match_minute(clock.video_to_utc(30 + 47 * 60)) == 45.0  # stoppage clamps
    assert clock.match_minute(clock.game_time_to_utc(*parse_game_time("2 - 22:00"))) == pytest.approx(67.0)
