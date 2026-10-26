import numpy as np
import pandas as pd
import pytest

from vision.analytics.heatmap import HeatmapConfig, compute_heatmap
from vision.analytics.movement import MovementConfig, compute_movement_metrics, direction_changes
from vision.pose.landmarks import L_FOOT, L_HEEL, R_FOOT, R_HEEL
from vision.trajectory.filtering import TrajectoryConfig, process_trajectory, sample_weights
from vision.trajectory.position import foot_contact_points


def traj_from(t, x, y, conf=None, **cfg):
    conf = np.full(len(t), 0.9) if conf is None else conf
    return process_trajectory(t, x, y, conf, TrajectoryConfig(**cfg))


def test_constant_velocity_speed_and_distance():
    t = np.arange(0, 4, 1 / 15)
    x = 1.0 + 1.5 * t  # 1.5 m/s along x
    y = np.full_like(t, 3.0)
    tr = traj_from(t, x, y)
    # The Kalman smoother starts from zero velocity, so allow a short start-up transient.
    assert tr["speed"].iloc[8:-3].to_numpy() == pytest.approx(1.5, abs=0.01)
    assert tr["speed"].iloc[3:-3].to_numpy() == pytest.approx(1.5, abs=0.05)
    m, _, _ = compute_movement_metrics(tr)
    assert m["distance"]["total_m"] == pytest.approx(1.5 * (t[-1] - t[0]), rel=0.005)


def test_square_path_distance():
    # Walk the perimeter of a 2 m square at 1 m/s → 8 m.
    fps = 30
    pts = []
    corners = [(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)]
    for (x0, y0), (x1, y1) in zip(corners[:-1], corners[1:], strict=True):
        n = 2 * fps
        for k in range(n):
            pts.append((x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n))
    pts.append(corners[-1])
    pts = np.array(pts)
    t = np.arange(len(pts)) / fps
    raw = traj_from(t, pts[:, 0], pts[:, 1], smoother="savgol", smoothing_window_s=0.0)  # no smoothing
    m, _, _ = compute_movement_metrics(raw)
    assert m["distance"]["total_m"] == pytest.approx(8.0, abs=0.02)
    smooth = traj_from(t, pts[:, 0], pts[:, 1])  # Kalman: cuts the four corners a little
    m2, _, _ = compute_movement_metrics(smooth)
    assert m2["distance"]["total_m"] == pytest.approx(8.0, rel=0.03)


def test_short_gaps_interpolated_long_gaps_not():
    t = np.arange(0, 6, 1 / 15)
    x = 1 + 0.5 * t
    y = np.full_like(t, 3.0)
    x[30:34] = np.nan  # 0.27 s gap → interpolated
    x[60:75] = np.nan  # 1.0 s gap → stays missing
    tr = traj_from(t, x, y)
    assert tr["interpolated"].iloc[30:34].all()
    assert tr["court_x"].iloc[30:34].notna().all()
    assert tr["court_x"].iloc[61:74].isna().all()
    assert tr["segment"].iloc[59] != tr["segment"].iloc[76]
    m, _, _ = compute_movement_metrics(tr)
    # No distance accumulated across the long gap (it would add 0.5 m/s × 1 s).
    expected = 0.5 * ((t[59] - t[0]) + (t[-1] - t[75]))
    assert m["distance"]["total_m"] == pytest.approx(expected, abs=0.02)
    assert m["data_quality"]["segments"] == 2


def test_outlier_and_speed_gate():
    t = np.arange(0, 3, 1 / 15)
    x = np.full_like(t, 2.0)
    y = np.full_like(t, 3.0)
    x[20] = 4.5  # 2.5 m jump for one frame: impossible
    tr = traj_from(t, x, y)
    assert not tr["raw_valid"].iloc[20]
    assert tr["reject_reason"].iloc[20] in ("outlier", "speed_gate")
    assert tr["court_x"].iloc[20] == pytest.approx(2.0, abs=1e-6)  # filled by short-gap interpolation
    assert tr["speed"].max() < 0.1


def test_out_of_bounds_and_low_confidence_rejected():
    t = np.arange(0, 1, 0.1)
    x = np.full(10, 2.0)
    y = np.full(10, 3.0)
    x[3] = 20.0
    conf = np.full(10, 0.9)
    conf[5] = 0.05
    tr = traj_from(t, x, y, conf)
    assert tr["reject_reason"].iloc[3] == "out_of_bounds"
    assert tr["reject_reason"].iloc[5] == "low_confidence"
    assert tr["court_x_raw"].iloc[3] == 20.0  # raw data is preserved


def test_noise_does_not_inflate_max_speed(rng):
    t = np.arange(0, 10, 1 / 15)
    x = 3.0 + rng.normal(0, 0.03, len(t))  # standing still, 3 cm jitter
    y = 3.0 + rng.normal(0, 0.03, len(t))
    tr = traj_from(t, x, y)
    m, events, _ = compute_movement_metrics(tr)
    assert m["speed"]["p95_mps"] < 0.5
    assert m["speed"]["moving_share"] < 0.05
    assert m["direction_changes"] == 0


def test_direction_changes_counted_on_zigzag_only():
    fps = 30
    t = np.arange(0, 6, 1 / fps)
    # Shuttle run: 2 m left/right at ~2 m/s → a change every ~1 s.
    phase = (t % 2.0) / 2.0
    x = 2.0 + np.where(phase < 0.5, phase * 4, (1 - phase) * 4)
    y = np.full_like(t, 3.0)
    tr = traj_from(t, x, y)
    ev = direction_changes(
        t,
        tr["court_x"].to_numpy(),
        tr["court_y"].to_numpy(),
        tr["vx"].to_numpy(),
        tr["vy"].to_numpy(),
        tr["segment"].to_numpy(),
        MovementConfig(),
    )
    assert 4 <= len(ev) <= 6
    assert all(e["angle_deg"] > 150 for e in ev)


def test_sample_weights_ignore_gaps_and_irregular_rate():
    t = np.array([0.0, 0.1, 0.2, 0.25, 0.3, 2.0, 2.1])
    valid = np.array([True, True, True, True, True, True, True])
    seg = np.array([0, 0, 0, 0, 0, 1, 1])
    w = sample_weights(t, valid, seg)
    assert w[4] <= 0.15 + 1e-9  # last sample of a segment is capped, not stretched over the gap
    assert w.sum() < 0.8


def test_heatmap_time_weighting_is_rate_independent():
    cfg = HeatmapConfig(cell_m=0.25, sigma_m=0.0)

    def hm_for(fps):
        t = np.arange(0, 10, 1 / fps)
        x = np.where(t < 7, 1.0, 5.0)  # 7 s left, 3 s right
        y = np.full_like(t, 3.0)
        seg = np.zeros(len(t), int)
        return compute_heatmap(x, y, sample_weights(t, np.ones(len(t), bool), seg), cfg)

    a, b = hm_for(15), hm_for(60)
    assert a.total_s == pytest.approx(10.0, abs=0.1)
    assert b.total_s == pytest.approx(10.0, abs=0.05)
    np.testing.assert_allclose(a.share.sum(), 1.0)
    np.testing.assert_allclose(a.share, b.share, atol=0.01)
    left = a.share[:, a.x_edges[:-1] < 3].sum()
    assert left == pytest.approx(0.7, abs=0.01)


def test_heatmap_smoothing_preserves_mass():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 6.1, 500)
    y = rng.uniform(0, 13.4, 500)
    w = np.full(500, 1 / 15)
    raw = compute_heatmap(x, y, w, HeatmapConfig(sigma_m=0.0))
    sm = compute_heatmap(x, y, w, HeatmapConfig(sigma_m=0.5))
    assert sm.seconds.sum() == pytest.approx(raw.seconds.sum(), rel=1e-9)
    assert sm.seconds.max() < raw.seconds.max()


def test_zone_time_and_side_detection():
    t = np.arange(0, 10, 0.1)
    x = np.where(t < 5, 1.0, 5.0)  # near side: player's left then right
    y = np.where(t < 5, 6.0, 1.0)  # front then rear
    tr = traj_from(t, x, y, max_speed_mps=100.0, outlier_min_residual_m=100.0)
    m, _, derived = compute_movement_metrics(tr)
    zt = m["zones"]["time_s"]
    assert zt["front_left"] == pytest.approx(5.0, abs=0.3)
    assert zt["rear_right"] == pytest.approx(5.0, abs=0.3)
    assert m["side_majority"] == 0
    assert m["zones"]["visited_pct"] == pytest.approx(100 * 2 / 9, abs=12)  # transition may touch a zone


def test_composite_score_is_labelled_and_can_be_disabled():
    t = np.arange(0, 20, 1 / 15)
    x = 3.05 + 1.5 * np.sin(t)
    y = 3.4 + 1.5 * np.cos(0.7 * t)
    tr = traj_from(t, x, y)
    m, _, _ = compute_movement_metrics(tr)
    assert m["composite"]["experimental"] is True
    assert "formula" in m["composite"] and m["composite"]["weights"]
    m2, _, _ = compute_movement_metrics(tr, MovementConfig(composite_enabled=False))
    assert m2["composite"] is None


def test_foot_contact_prefers_pose_feet_and_falls_back_to_box():
    T = 3
    xy = np.full((T, 33, 2), np.nan)
    vis = np.zeros((T, 33))
    bbox = np.array([[100, 100, 160, 300]] * 3, float)
    xy[0] = 130
    xy[0, L_HEEL] = (120, 290)
    xy[0, L_FOOT] = (124, 295)
    xy[0, R_HEEL] = (140, 290)
    xy[0, R_FOOT] = (144, 295)
    vis[0, [L_HEEL, L_FOOT, R_HEEL, R_FOOT]] = 0.9
    bbox[2] = np.nan
    df = foot_contact_points(xy, vis, bbox, np.array([0.9, 0.9, np.nan]))
    assert df["position_source"].tolist() == ["pose_feet", "bbox", "none"]
    assert df.loc[0, "img_x"] == pytest.approx(132)
    assert df.loc[1, "img_y"] == pytest.approx(300)
    assert df.loc[0, "position_confidence"] > df.loc[1, "position_confidence"]
    assert np.isnan(df.loc[2, "img_x"])


def test_pose_feet_outside_box_rejected():
    xy = np.full((1, 33, 2), 150.0)
    vis = np.full((1, 33), 0.9)
    xy[0, [L_HEEL, L_FOOT, R_HEEL, R_FOOT]] = (900, 900)  # pose drifted to someone else
    df = foot_contact_points(xy, vis, np.array([[100, 100, 160, 300]], float), np.array([0.9]))
    assert df["position_source"].iloc[0] == "bbox"


def test_metrics_empty_trajectory_is_safe():
    t = np.arange(0, 2, 0.1)
    tr = traj_from(t, np.full(20, np.nan), np.full(20, np.nan))
    m, events, _ = compute_movement_metrics(tr)
    assert m["distance"]["total_m"] == 0
    assert m["speed"]["mean_mps"] is None
    assert m["observed_s"] == 0
    assert isinstance(pd.DataFrame(events), pd.DataFrame)


def test_kalman_handles_uneven_sampling_and_noise(rng):
    # 25 fps footage sampled at a non-divisor rate alternates 0.04 / 0.08 s steps.
    t = np.cumsum(np.tile([0.04, 0.08], 100))
    x = 1.0 + 1.0 * t
    y = np.full_like(t, 3.0)
    noisy_x = x + rng.normal(0, 0.1, len(t))
    noisy_y = y + rng.normal(0, 0.1, len(t))
    tr = traj_from(t, noisy_x, noisy_y)
    v = tr["speed"].iloc[10:-10]
    assert v.median() == pytest.approx(1.0, abs=0.15)
    assert v.max() < 2.0  # 10 cm jitter must not become sprint speeds
    assert np.nanmean(np.abs(tr["court_x"] - x)) < 0.06


def test_persistent_jump_starts_new_segment():
    t = np.arange(0, 4, 0.08)
    x = np.full_like(t, 3.0)
    y = np.where(t < 2, 3.0, 11.0)  # identity swap to the far player that the tracker did not catch
    tr = traj_from(t, x, y)
    seg = tr["segment"].to_numpy()
    assert seg[t < 1.9].max() != seg[t > 2.5].min()  # never connected
    assert tr["speed"].max() < 1.0  # no 100 m/s "sprint" across the net
