import numpy as np
import pytest

from vision.evaluation.metrics import (
    classification_metrics,
    detection_pr,
    event_detection_prf,
    mot_metrics,
    position_errors,
    target_tracking_accuracy,
)
from vision.pose.landmarks import L_HIP, L_SHOULDER, NOSE, R_HIP, R_SHOULDER, R_WRIST, body_metrics
from vision.strokes.classifier import FEATURE_NAMES, event_features, expected_calibration_error, train_stroke_model
from vision.strokes.detection import StrokeConfig, detect_strokes


def standing_pose(T: int) -> tuple[np.ndarray, np.ndarray]:
    xy = np.zeros((T, 33, 2))
    xy[:, :] = (300, 300)
    xy[:, NOSE] = (300, 120)
    xy[:, L_SHOULDER] = (270, 160)
    xy[:, R_SHOULDER] = (330, 160)
    xy[:, L_HIP] = (285, 260)
    xy[:, R_HIP] = (315, 260)
    xy[:, R_WRIST] = (360, 230)
    for i in range(33):
        xy[:, i] += np.random.default_rng(i).normal(0, 0.3, (T, 2))
    return xy, np.full((T, 33), 0.95)


def test_detects_wrist_swings_and_not_noise():
    fps = 30
    T = 300
    t = np.arange(T) / fps
    xy, vis = standing_pose(T)
    swings = [2.0, 5.0, 8.0]
    for s in swings:
        # Slow preparation lifting the racket hand overhead, then a fast forward swing near contact.
        prep = (t > s - 0.6) & (t <= s - 0.08)
        ph = (t[prep] - (s - 0.6)) / 0.52
        xy[prep, R_WRIST, 0] = 360 + 20 * ph
        xy[prep, R_WRIST, 1] = 230 - 200 * ph
        swing = (t > s - 0.08) & (t <= s + 0.08)
        ph = (t[swing] - (s - 0.08)) / 0.16
        xy[swing, R_WRIST, 0] = 380 - 160 * ph
        xy[swing, R_WRIST, 1] = 30 + 60 * ph
        back = (t > s + 0.08) & (t <= s + 0.6)
        ph = (t[back] - (s + 0.08)) / 0.52
        xy[back, R_WRIST, 0] = 220 + 140 * ph
        xy[back, R_WRIST, 1] = 90 + 140 * ph
    events, diag = detect_strokes(t, xy, vis, StrokeConfig(handedness="right"))
    assert diag["handedness"] == "right"
    found = [e["t"] for e in events]
    assert event_detection_prf(swings, found, 0.15)["f1"] == pytest.approx(1.0)
    assert all(e["experimental"] for e in events)
    assert all(e["height_class"] == "overhead" for e in events)
    assert all(e["shot_type"] == "overhead_unspecified" for e in events)  # never claims smash/clear/drop

    still, vis2 = standing_pose(T)
    events2, _ = detect_strokes(t, still, vis2)
    assert events2 == []


def test_body_metrics_shapes_and_nans():
    xy, vis = standing_pose(10)
    vis[3] = 0.0
    m = body_metrics(xy, vis)
    assert m["torso_length_px"].shape == (10,)
    assert np.isnan(m["right_elbow_angle"][3])
    assert np.isfinite(m["stance_width"][0]) or np.isnan(m["stance_width"][0])


def test_classifier_training_uses_grouped_cv():
    rng = np.random.default_rng(0)
    n = 120
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    y = np.where(X[:, 1] > 0, "clear", "drive")
    X[rng.random((n, len(FEATURE_NAMES))) < 0.05] = np.nan
    groups = [f"s{i % 4}" for i in range(n)]
    model, report = train_stroke_model(X, list(y), groups, target="shot_type")
    assert report["n_sessions"] == 4
    assert "GroupKFold" in report["cv"]
    assert 0.0 <= report["macro_f1"] <= 1.0
    assert report["macro_f1"] > 0.7  # learnable synthetic signal
    preds = model.predict(X[:5], min_confidence=0.99)
    assert all(p["label"] in ("clear", "drive", "uncertain") for p in preds)
    assert model.card["evaluation"]["confusion_matrix"]["labels"] == ["clear", "drive"]


def test_classifier_refuses_single_session():
    X = np.zeros((10, len(FEATURE_NAMES)))
    with pytest.raises(ValueError, match="at least 2 sessions"):
        train_stroke_model(X, ["a"] * 5 + ["b"] * 5, ["s1"] * 10, target="shot_type")


def test_event_features_shape():
    xy, vis = standing_pose(60)
    f = event_features(np.arange(60) / 30, xy, vis, 1.0)
    assert f.shape == (len(FEATURE_NAMES),)


def test_event_prf_tolerance():
    r = event_detection_prf([1.0, 2.0, 3.0], [1.1, 2.4, 3.0, 5.0], 0.25)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 2, 1)
    assert r["precision"] == pytest.approx(0.5)
    assert r["recall"] == pytest.approx(2 / 3)


def test_detection_pr():
    gt = {0: [("a", np.array([0, 0, 10, 10]))], 1: [("a", np.array([0, 0, 10, 10]))]}
    pred = {0: [(None, np.array([1, 1, 10, 10]), 0.9), (None, np.array([50, 50, 60, 60]), 0.5)], 1: []}
    r = detection_pr(gt, pred)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1)


def test_mot_metrics_perfect_and_switch():
    b1, b2 = np.array([0, 0, 10, 10]), np.array([100, 0, 110, 10])
    gt = {f: [("A", b1), ("B", b2)] for f in range(10)}
    perfect = {f: [(1, b1), (2, b2)] for f in range(10)}
    r = mot_metrics(gt, perfect)
    assert r["idf1"] == pytest.approx(1.0) and r["id_switches"] == 0 and r["mota"] == pytest.approx(1.0)
    switched = {f: [(1, b1), (2, b2)] if f < 5 else [(2, b1), (1, b2)] for f in range(10)}
    r2 = mot_metrics(gt, switched)
    assert r2["id_switches"] == 2
    assert r2["idf1"] == pytest.approx(0.5)


def test_target_accuracy_and_position_errors():
    gt = {0: np.array([0, 0, 10, 10]), 1: np.array([0, 0, 10, 10]), 2: np.array([0, 0, 10, 10])}
    pred = {0: np.array([0, 0, 10, 10]), 1: np.array([50, 50, 60, 60])}
    r = target_tracking_accuracy(gt, pred)
    assert r["correct_share"] == pytest.approx(1 / 3) and r["missing_share"] == pytest.approx(1 / 3)
    e = position_errors(np.array([[0, 0], [1, 1], [2, 2]]), np.array([[0, 0.1], [1, 1.2], [np.nan, 2]]))
    assert e["n"] == 2 and e["mae_m"] == pytest.approx(0.15)


def test_classification_metrics_and_ece():
    r = classification_metrics(["a", "a", "b"], ["a", "b", "b"])
    assert r["accuracy"] == pytest.approx(2 / 3)
    assert r["confusion_matrix"]["matrix"] == [[1, 1], [0, 1]]
    proba = np.array([[0.9, 0.1], [0.6, 0.4]])
    assert expected_calibration_error(proba, np.array([0, 1])) == pytest.approx(0.5 * 0.1 + 0.5 * 0.6)
