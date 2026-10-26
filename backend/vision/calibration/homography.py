"""Planar homography between image pixels and court metres.

The court floor is a plane, so a 3×3 homography ``H`` maps image points of floor features
(painted lines, the player's feet) to court coordinates. ``H`` is only valid for points on
the floor: projecting a hip or a head through it gives meaningless positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import cv2
import numpy as np

from vision.geometry.court import COURT_LENGTH, COURT_WIDTH, keypoint_by_id

# Acceptance thresholds. A homography fitted from exactly four points always has zero
# residual, so for n == 4 we rely on the conditioning checks instead.
MAX_MEAN_ERROR_M = 0.20
MAX_POINT_ERROR_M = 0.50
MAX_CONDITION = 1e7
MIN_TRIANGLE_AREA_M2 = 0.25  # court-space area below which three points count as collinear
MIN_TRIANGLE_AREA_PX2 = 100.0
RANSAC_MIN_POINTS = 6
RANSAC_THRESHOLD_M = 0.15


class CalibrationError(ValueError):
    """The supplied correspondences cannot produce a trustworthy homography."""


@dataclass
class Calibration:
    H: np.ndarray  # image px -> court metres
    image_points: np.ndarray
    court_points: np.ndarray
    point_ids: list[str]
    inliers: np.ndarray
    method: str
    image_size: tuple[int, int] | None = None  # (width, height) the points refer to
    errors_m: np.ndarray = field(default_factory=lambda: np.zeros(0))
    errors_px: np.ndarray = field(default_factory=lambda: np.zeros(0))
    condition_number: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def H_inv(self) -> np.ndarray:
        return np.linalg.inv(self.H)

    @property
    def mean_error_m(self) -> float:
        e = self.errors_m[self.inliers] if self.errors_m.size else self.errors_m
        return float(np.mean(e)) if e.size else 0.0

    @property
    def max_error_m(self) -> float:
        e = self.errors_m[self.inliers] if self.errors_m.size else self.errors_m
        return float(np.max(e)) if e.size else 0.0

    @property
    def mean_error_px(self) -> float:
        e = self.errors_px[self.inliers] if self.errors_px.size else self.errors_px
        return float(np.mean(e)) if e.size else 0.0

    def to_dict(self) -> dict:
        return {
            "homography": self.H.tolist(),
            "image_points": self.image_points.tolist(),
            "court_points": self.court_points.tolist(),
            "point_ids": list(self.point_ids),
            "inliers": self.inliers.astype(bool).tolist(),
            "method": self.method,
            "image_size": list(self.image_size) if self.image_size else None,
            "errors_m": self.errors_m.tolist(),
            "errors_px": self.errors_px.tolist(),
            "mean_error_m": self.mean_error_m,
            "max_error_m": self.max_error_m,
            "mean_error_px": self.mean_error_px,
            "condition_number": self.condition_number,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Calibration:
        return cls(
            H=np.asarray(d["homography"], float),
            image_points=np.asarray(d["image_points"], float).reshape(-1, 2),
            court_points=np.asarray(d["court_points"], float).reshape(-1, 2),
            point_ids=list(d.get("point_ids", [])),
            inliers=np.asarray(d.get("inliers", [True] * len(d["image_points"])), bool),
            method=d.get("method", "manual"),
            image_size=tuple(d["image_size"]) if d.get("image_size") else None,
            errors_m=np.asarray(d.get("errors_m", []), float),
            errors_px=np.asarray(d.get("errors_px", []), float),
            condition_number=float(d.get("condition_number", 0.0)),
            warnings=list(d.get("warnings", [])),
        )

    def scaled_to(self, width: int, height: int) -> np.ndarray:
        """Homography for frames resized to ``(width, height)`` from the calibration size."""
        if not self.image_size:
            return self.H
        sx = self.image_size[0] / width
        sy = self.image_size[1] / height
        S = np.diag([sx, sy, 1.0])  # resized px -> calibration px
        return self.H @ S


def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a homography to (N, 2) points. Points mapping to infinity become NaN."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    if pts.size == 0:
        return pts.copy()
    homog = np.hstack([pts, np.ones((len(pts), 1))]) @ np.asarray(H, float).T
    w = homog[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = homog[:, :2] / w
    out[np.abs(w[:, 0]) < 1e-12] = np.nan
    return out


def _min_triangle_area(pts: np.ndarray) -> float:
    best = np.inf
    for i, j, k in combinations(range(len(pts)), 3):
        a, b, c = pts[i], pts[j], pts[k]
        area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
        best = min(best, area)
    return float(best)


def _largest_triangle_area(pts: np.ndarray) -> float:
    best = 0.0
    for i, j, k in combinations(range(len(pts)), 3):
        a, b, c = pts[i], pts[j], pts[k]
        best = max(best, 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])))
    return best


def _check_geometry(image_pts: np.ndarray, court_pts: np.ndarray) -> None:
    n = len(image_pts)
    if n < 4:
        raise CalibrationError(f"need at least 4 point correspondences, got {n}")
    if len(court_pts) != n:
        raise CalibrationError("image and court point counts differ")
    if not (np.all(np.isfinite(image_pts)) and np.all(np.isfinite(court_pts))):
        raise CalibrationError("points contain NaN or infinity")
    if len(np.unique(np.round(court_pts, 3), axis=0)) < n:
        raise CalibrationError("the same court landmark is used more than once")
    if len(np.unique(np.round(image_pts, 1), axis=0)) < n:
        raise CalibrationError("two landmarks were placed on the same image pixel")
    # With four points every triple must be non-collinear; with more, the set must not lie on
    # a line overall (some collinear triples are normal, e.g. points along one baseline).
    if n == 4:
        if _min_triangle_area(court_pts) < MIN_TRIANGLE_AREA_M2:
            raise CalibrationError("three of the four court landmarks are collinear; pick a spread-out set")
        if _min_triangle_area(image_pts) < MIN_TRIANGLE_AREA_PX2:
            raise CalibrationError("three of the four image points are (nearly) on one line")
    else:
        if _largest_triangle_area(court_pts) < 1.0:
            raise CalibrationError("court landmarks are (nearly) collinear")
        if _largest_triangle_area(image_pts) < 400.0:
            raise CalibrationError("image points are (nearly) collinear")


def _normalised_condition(H: np.ndarray, image_pts: np.ndarray) -> float:
    """Condition number of H after Hartley normalisation of the image points.

    A raw pixel→metre homography is badly scaled by construction, so its condition number says
    little. Normalising the input space makes the number comparable across resolutions.
    """
    c = image_pts.mean(axis=0)
    s = np.sqrt(2) / max(np.mean(np.linalg.norm(image_pts - c, axis=1)), 1e-9)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    Hn = H @ np.linalg.inv(T)
    Hn = Hn / np.linalg.norm(Hn)
    return float(np.linalg.cond(Hn))


def fit_homography(
    image_points: np.ndarray | list,
    court_points: np.ndarray | list | None = None,
    *,
    point_ids: list[str] | None = None,
    method: str = "manual",
    image_size: tuple[int, int] | None = None,
) -> Calibration:
    """Fit and validate an image→court homography.

    Either pass ``court_points`` directly or ``point_ids`` naming court keypoints. With six or
    more correspondences RANSAC rejects mis-clicked landmarks; with four or five a direct least
    squares fit is used. Raises :class:`CalibrationError` when the result is unusable.
    """
    img = np.asarray(image_points, float).reshape(-1, 2)
    if court_points is None:
        if not point_ids:
            raise CalibrationError("provide court_points or point_ids")
        try:
            court = np.array([[keypoint_by_id(p).x, keypoint_by_id(p).y] for p in point_ids], float)
        except KeyError as exc:
            raise CalibrationError(str(exc)) from exc
    else:
        court = np.asarray(court_points, float).reshape(-1, 2)
    ids = list(point_ids) if point_ids else [f"p{i}" for i in range(len(img))]

    _check_geometry(img, court)

    n = len(img)
    if n >= RANSAC_MIN_POINTS:
        H, mask = cv2.findHomography(img, court, cv2.RANSAC, RANSAC_THRESHOLD_M, maxIters=5000, confidence=0.999)
        inliers = mask.ravel().astype(bool) if mask is not None else np.ones(n, bool)
        if inliers.sum() < max(5, int(np.ceil(0.6 * n))):
            # Fewer than five agreeing points cannot be verified (any four points fit exactly), and a small
            # consensus means several landmarks disagree: more likely a wrong assignment.
            raise CalibrationError(
                f"only {int(inliers.sum())} of {n} landmarks agree on one court plane; "
                "check that each point is assigned to the right court landmark"
            )
        if H is not None and inliers.sum() >= 4:
            # Refit on inliers with plain least squares for the final estimate.
            H2, _ = cv2.findHomography(img[inliers], court[inliers], 0)
            H = H2 if H2 is not None else H
    else:
        H, _ = cv2.findHomography(img, court, 0)
        inliers = np.ones(n, bool)
    if H is None or not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-15:
        raise CalibrationError("homography estimation failed (degenerate point configuration)")
    H = H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H

    warnings: list[str] = []
    cond = _normalised_condition(H, img[inliers])
    if cond > MAX_CONDITION:
        raise CalibrationError(f"homography is ill-conditioned (condition number {cond:.2e})")

    # The image points must all lie on the same side of the horizon (w > 0 after mapping),
    # otherwise the mapping folds the plane and is physically impossible.
    w = (np.hstack([img, np.ones((n, 1))]) @ H.T)[:, 2]
    if not (np.all(w > 0) or np.all(w < 0)):
        raise CalibrationError("points straddle the horizon line; check the landmark assignments")

    mapped = project(H, img)
    errors_m = np.linalg.norm(mapped - court, axis=1)
    back = project(np.linalg.inv(H), court)
    errors_px = np.linalg.norm(back - img, axis=1)

    cal = Calibration(
        H=H,
        image_points=img,
        court_points=court,
        point_ids=ids,
        inliers=inliers,
        method=method,
        image_size=image_size,
        errors_m=errors_m,
        errors_px=errors_px,
        condition_number=cond,
        warnings=warnings,
    )
    if (~inliers).any():
        rejected = [ids[i] for i in np.flatnonzero(~inliers)]
        warnings.append(f"RANSAC rejected {len(rejected)} landmark(s) as outliers: {', '.join(rejected)}")
    if n == 4:
        warnings.append(
            "exactly 4 points: the fit is exact, so reprojection error cannot reveal mistakes. "
            "Add more landmarks for a verifiable calibration."
        )
    if cal.mean_error_m > MAX_MEAN_ERROR_M:
        raise CalibrationError(
            f"mean reprojection error {cal.mean_error_m:.2f} m exceeds {MAX_MEAN_ERROR_M} m; "
            "some landmarks are probably assigned to the wrong court point"
        )
    if cal.max_error_m > MAX_POINT_ERROR_M:
        warnings.append(f"largest landmark error is {cal.max_error_m:.2f} m")

    # Warn if the visible part of the court is small: extrapolation far beyond the landmarks
    # magnifies small click errors.
    hull_area = cv2.contourArea(cv2.convexHull(court[inliers].astype(np.float32).reshape(-1, 1, 2)))
    if hull_area < 0.15 * COURT_WIDTH * COURT_LENGTH / 2:
        warnings.append("landmarks cover a small part of the court; positions far from them are less accurate")
    return cal


def court_to_image_polyline(H_inv: np.ndarray, start, end, n: int = 32) -> np.ndarray:
    """Sample a court-space segment and project it into the image (for overlays)."""
    t = np.linspace(0, 1, n)[:, None]
    seg = np.asarray(start, float) + t * (np.asarray(end, float) - np.asarray(start, float))
    return project(H_inv, seg)


def calibration_from_camera(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Exact image→court homography for a pinhole camera looking at the z = 0 court plane.

    Used by the synthetic scene generator and tests: ``x_img ~ K [r1 r2 t] [X Y 1]^T``.
    """
    Hc = K @ np.column_stack([R[:, 0], R[:, 1], t.reshape(3)])  # court -> image
    H = np.linalg.inv(Hc)
    return H / H[2, 2]
