import numpy as np
import pytest

from varflash.vision.homography import fit_homography, from_keypoints
from varflash.vision.offside import judge_offside
from varflash.vision.pitch import KEYPOINTS
from varflash.vision.teams import assign_teams, hex_to_lab


def _synthetic_camera():
    """A plausible broadcast-camera homography pitch -> image."""
    return np.array([[12.0, -4.0, 300.0], [0.5, 6.0, 120.0], [0.0, -0.004, 1.0]])


def _project(H, pts):
    pts = np.c_[np.asarray(pts, float), np.ones(len(pts))]
    out = (H @ pts.T).T
    return out[:, :2] / out[:, 2:]


def test_homography_recovers_pitch_coordinates():
    G = _synthetic_camera()
    names = ["lbox_top", "lbox_bottom", "halfway_top", "halfway_bottom", "lpen_spot", "center_spot"]
    image_pts = _project(G, [KEYPOINTS[n] for n in names])
    h = from_keypoints(dict(zip(names, map(tuple, image_pts))))
    assert h.reprojection_error_m < 1e-4

    pitch = np.array([[30.0, 20.0], [40.0, 50.0]])
    np.testing.assert_allclose(h.to_pitch(_project(G, pitch)), pitch, atol=1e-6)
    np.testing.assert_allclose(h.to_image(pitch), _project(G, pitch), atol=1e-6)


def test_homography_needs_four_points():
    with pytest.raises(ValueError):
        fit_homography([[0, 0], [1, 0], [0, 1]], [[0, 0], [1, 0], [0, 1]])


def test_offside_basic_attacking_right():
    call = judge_offside(attacker_x=[80.0, 70.0], defender_x=[79.0, 75.0, 60.0], ball_x=65.0)
    assert call.offside
    assert call.margin_m == pytest.approx(1.0)
    assert call.line_x == pytest.approx(79.0)
    assert call.probability > 0.95


def test_level_is_onside_and_probability_half():
    call = judge_offside([79.0], [79.0, 70.0], ball_x=60.0)
    assert not call.offside
    assert call.probability == pytest.approx(0.5)


def test_attacking_left_mirrors():
    right = judge_offside([80.0], [79.0, 70.0], ball_x=60.0, attack_right=True)
    left = judge_offside([25.0], [26.0, 35.0], ball_x=45.0, attack_right=False)
    assert left.offside and left.margin_m == pytest.approx(right.margin_m)
    assert left.line_x == pytest.approx(26.0)


def test_cannot_be_offside_in_own_half_or_behind_ball():
    assert not judge_offside([50.0], [40.0], ball_x=30.0).offside
    assert not judge_offside([80.0], [70.0], ball_x=85.0).offside


def test_keeper_visible_uses_second_last_defender():
    # Keeper at 100, last outfielder at 82: line is 82 when the keeper is in shot.
    call = judge_offside([81.0], [100.0, 82.0, 70.0], ball_x=60.0, keeper_off_screen=False)
    assert not call.offside and call.line_x == pytest.approx(82.0)


def test_team_assignment_by_colour():
    rng = np.random.default_rng(0)
    sky_blue, red, yellow = hex_to_lab("#75AADB"), hex_to_lab("#D52B1E"), hex_to_lab("#F5E642")
    colors = np.vstack([
        sky_blue + rng.normal(0, 2, (8, 3)),
        red + rng.normal(0, 2, (8, 3)),
        yellow + rng.normal(0, 2, (1, 3)),
    ])
    labels = assign_teams(colors, attacking_color="#D52B1E")
    assert (labels[8:16] == "A").all() and (labels[:8] == "B").all()
    assert labels[16] == "other"
