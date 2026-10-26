import numpy as np
import pytest

from vision.calibration.auto import auto_calibrate
from vision.calibration.homography import Calibration, CalibrationError, fit_homography, project
from vision.geometry.court import keypoint_by_id
from vision.synthetic.scene import court_keypoint_pixels, default_camera, make_scene

IDS = [
    "near_baseline__left_doubles",
    "near_baseline__right_doubles",
    "far_baseline__left_doubles",
    "far_baseline__right_doubles",
    "near_short_service__left_singles",
    "far_short_service__right_singles",
    "near_long_service__center",
    "far_long_service__left_singles",
]
GRID = np.array([[x, y] for x in np.linspace(0, 6.1, 7) for y in np.linspace(0, 13.4, 9)])


def _true_pixels(cam, ids):
    return court_keypoint_pixels(cam, ids)


def test_exact_recovery_from_camera():
    cam = default_camera(1280, 720)
    cal = fit_homography(_true_pixels(cam, IDS), point_ids=IDS, image_size=(1280, 720))
    assert cal.mean_error_m < 1e-6
    uv = project(np.linalg.inv(cam.homography), GRID)
    np.testing.assert_allclose(project(cal.H, uv), GRID, atol=1e-6)


def test_coordinate_round_trip():
    cam = default_camera(1280, 720)
    H = cam.homography
    pts = np.array([[1.0, 2.0], [3.05, 6.7], [5.5, 12.0]])
    back = project(H, project(np.linalg.inv(H), pts))
    np.testing.assert_allclose(back, pts, atol=1e-9)


def test_noise_gives_small_error(rng):
    cam = default_camera(1280, 720)
    px = _true_pixels(cam, IDS) + rng.normal(0, 1.0, (len(IDS), 2))
    cal = fit_homography(px, point_ids=IDS)
    est = project(cal.H, project(np.linalg.inv(cam.homography), GRID))
    assert np.linalg.norm(est - GRID, axis=1).mean() < 0.06
    assert cal.mean_error_m < 0.05


def test_ransac_rejects_misclicked_landmark():
    cam = default_camera(1280, 720)
    px = _true_pixels(cam, IDS)
    px[2] += [60, -25]  # one wrong click
    cal = fit_homography(px, point_ids=IDS)
    assert not cal.inliers[2]
    assert cal.inliers.sum() == len(IDS) - 1
    assert any("RANSAC rejected" in w for w in cal.warnings)


def test_needs_four_points():
    with pytest.raises(CalibrationError, match="at least 4"):
        fit_homography(np.zeros((3, 2)), court_points=np.zeros((3, 2)))


def test_collinear_court_points_rejected():
    ids = [
        "near_baseline__left_doubles",
        "near_baseline__left_singles",
        "near_baseline__center",
        "near_baseline__right_doubles",
    ]
    cam = default_camera(1280, 720)
    with pytest.raises(CalibrationError, match="collinear"):
        fit_homography(_true_pixels(cam, ids), point_ids=ids)


def test_duplicate_landmarks_rejected():
    with pytest.raises(CalibrationError):
        fit_homography([[0, 0], [10, 0], [0, 10], [10, 10]], court_points=[[0, 0], [1, 0], [0, 1], [0, 1]])


def test_same_pixel_rejected():
    with pytest.raises(CalibrationError, match="same image pixel"):
        fit_homography([[5, 5], [5, 5], [100, 0], [0, 100]], court_points=[[0, 0], [6.1, 0], [0, 13.4], [6.1, 13.4]])


def test_wrong_assignment_detected():
    cam = default_camera(1280, 720)
    px = _true_pixels(cam, IDS[:6])
    swapped = IDS[:6].copy()
    swapped[0], swapped[1] = swapped[1], swapped[0]
    swapped[2], swapped[4] = swapped[4], swapped[2]
    with pytest.raises(CalibrationError):
        fit_homography(px, point_ids=swapped)


def test_unknown_keypoint():
    with pytest.raises(CalibrationError, match="unknown court keypoint"):
        fit_homography(np.zeros((4, 2)), point_ids=["a", "b", "c", "d"])


def test_four_point_fit_warns_unverifiable():
    cam = default_camera(1280, 720)
    cal = fit_homography(_true_pixels(cam, IDS[:4]), point_ids=IDS[:4])
    assert any("exactly 4 points" in w for w in cal.warnings)


def test_scaled_homography_for_resized_frames():
    cam = default_camera(1280, 720)
    cal = fit_homography(_true_pixels(cam, IDS), point_ids=IDS, image_size=(1280, 720))
    H_half = cal.scaled_to(640, 360)
    p_full = _true_pixels(cam, ["near_short_service__left_singles"])
    np.testing.assert_allclose(project(H_half, p_full / 2), project(cal.H, p_full), atol=1e-6)


def test_serialisation_round_trip():
    cam = default_camera(1280, 720)
    cal = fit_homography(_true_pixels(cam, IDS), point_ids=IDS, image_size=(1280, 720))
    cal2 = Calibration.from_dict(cal.to_dict())
    np.testing.assert_allclose(cal.H, cal2.H)
    assert cal2.point_ids == IDS and cal2.image_size == (1280, 720)


def test_point_at_infinity_is_nan():
    H = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]], float)
    out = project(H, np.array([[0.0, 5.0]]))
    assert np.isnan(out).all()


@pytest.mark.parametrize("view", ["baseline", "corner"])
def test_auto_calibration_on_synthetic_frame(view):
    scene = make_scene(duration=2, fps=30, seed=4, view=view)
    res = auto_calibrate(scene.render(20))
    assert res.success, res.message
    est = project(res.calibration.H, project(np.linalg.inv(scene.camera.homography), GRID))
    assert np.linalg.norm(est - GRID, axis=1).mean() < 0.15
    assert keypoint_by_id(res.calibration.point_ids[0])  # ids are real court keypoints


def test_auto_calibration_fails_gracefully_without_court():
    blank = np.full((360, 640, 3), 90, np.uint8)
    res = auto_calibrate(blank)
    assert not res.success
    assert res.message


def test_auto_calibration_rejects_cluttered_non_court_photo():
    import os

    import cv2

    ultralytics = pytest.importorskip("ultralytics")
    for name in ("bus.jpg", "zidane.jpg"):
        img = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", name))
        assert not auto_calibrate(img).success, name
