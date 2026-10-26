import numpy as np
import pytest

from vision.geometry.court import (
    COURT_LENGTH,
    COURT_WIDTH,
    NET_Y,
    SIDE_ALLEY,
    SINGLES_WIDTH,
    ZoneConfig,
    assign_zones,
    court_keypoints,
    court_lines,
    from_half_coords,
    keypoint_by_id,
    to_half_coords,
)


def test_bwf_dimensions():
    assert COURT_LENGTH == pytest.approx(13.40)
    assert COURT_WIDTH == pytest.approx(6.10)
    assert SINGLES_WIDTH == pytest.approx(5.18)
    assert SIDE_ALLEY == pytest.approx(0.46)
    assert NET_Y == pytest.approx(6.70)
    # Short service line 1.98 m from the net; doubles long service line 0.76 m inside the baseline.
    assert keypoint_by_id("near_short_service__left_doubles").y == pytest.approx(4.72)
    assert keypoint_by_id("far_short_service__left_doubles").y == pytest.approx(8.68)
    assert keypoint_by_id("near_long_service__left_doubles").y == pytest.approx(0.76)
    assert keypoint_by_id("far_long_service__right_doubles").y == pytest.approx(12.64)


def test_keypoints_unique_and_on_court():
    kps = court_keypoints()
    ids = [k.id for k in kps]
    assert len(ids) == len(set(ids))
    assert len(kps) == 6 * 5 + 2
    for k in kps:
        assert 0 <= k.x <= COURT_WIDTH and 0 <= k.y <= COURT_LENGTH


def test_lines_include_all_markings():
    ids = {ln.id for ln in court_lines()}
    for needed in [
        "near_baseline",
        "far_baseline",
        "left_doubles",
        "right_doubles",
        "left_singles",
        "right_singles",
        "near_short_service",
        "far_short_service",
        "near_long_service",
        "far_long_service",
        "near_center",
        "far_center",
        "net",
    ]:
        assert needed in ids
    center = next(ln for ln in court_lines() if ln.id == "near_center")
    assert center.end[1] == pytest.approx(4.72)  # centre line stops at the short service line


@pytest.mark.parametrize("side", [0, 1])
def test_half_coordinates_round_trip(side):
    lat = np.array([0.0, 2.59, 5.18])
    dep = np.array([0.5, 3.0, 6.7])
    x, y = from_half_coords(lat, dep, side)
    lat2, dep2 = to_half_coords(x, y, side)
    np.testing.assert_allclose(lat, lat2)
    np.testing.assert_allclose(dep, dep2)


def test_player_relative_left_right():
    # Near-side player faces +y: their left is x = 0. Far-side player faces -y: their left is x = 6.1.
    lat_near, _ = to_half_coords(np.array([0.6]), np.array([3.0]), 0)
    lat_far, _ = to_half_coords(np.array([0.6]), np.array([10.0]), 1)
    assert lat_near[0] < 1.0
    assert lat_far[0] > 4.0


def test_zone_assignment():
    cfg = ZoneConfig()
    z = assign_zones(np.array([0.2, 2.6, 5.0, 2.6, np.nan, 9.0]), np.array([0.5, 3.0, 6.0, 6.69, 1.0, 3.0]), cfg)
    assert list(z) == ["front_left", "mid_center", "rear_right", "rear_center", None, "out"]


def test_zone_config_validation():
    with pytest.raises(ValueError):
        ZoneConfig(depth_edges=(0, 1, 2), depth_names=("a", "b", "c"))
    with pytest.raises(ValueError):
        ZoneConfig(lateral_edges=(0, 3, 2, 5.18))
