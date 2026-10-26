"""Automatic court detection: white-line extraction → Hough lines → model-fitting search.

The result is a *suggestion*: it is returned as named landmark correspondences so the user can
inspect and correct it in the manual calibration editor before analysis.

Algorithm
---------
1. Line mask: thin, bright, low-saturation structures (white top-hat ∧ HSV threshold).
2. Probabilistic Hough transform → segments, split into a "horizontal" family (baselines,
   service lines; nearly horizontal from a behind-the-baseline camera) and a "sideline" family.
3. Collinear segments are merged into lines, ranked by supporting length.
4. Hypothesis search: every (image horizontal pair × image sideline pair) is matched against
   every (model horizontal pair × model sideline pair) with consistent ordering. Each pairing
   gives four intersections and hence a homography; it is scored by projecting all model lines
   into the image and measuring how much of them lands on the line mask.
5. The best hypothesis is refined: each detected line is snapped to the closest projected model
   line, all matched intersections become correspondences, and a RANSAC homography is fitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from vision.calibration.homography import Calibration, CalibrationError, fit_homography, project
from vision.geometry.court import HORIZONTAL_LINES, VERTICAL_LINES, keypoint_by_id, sample_court_lines

_WORK_WIDTH = 960
_SIDE_MODEL = ["left_doubles", "left_singles", "right_singles", "right_doubles"]
_HORIZ_MODEL = list(HORIZONTAL_LINES)  # near -> far


@dataclass
class ImageLine:
    # Line as a*x + b*y + c = 0 with (a, b) unit normal, in working-image pixels.
    a: float
    b: float
    c: float
    support: float  # total length of merged segments
    family: str  # horizontal | side
    key: float  # sort key: y at image centre (horizontal) or x at mid-height (side)

    def intersect(self, other: ImageLine) -> np.ndarray | None:
        det = self.a * other.b - other.a * self.b
        if abs(det) < 1e-9:
            return None
        x = (self.b * other.c - other.b * self.c) / det
        y = (other.a * self.c - self.a * other.c) / det
        return np.array([x, y])

    def distance(self, pts: np.ndarray) -> np.ndarray:
        return np.abs(self.a * pts[:, 0] + self.b * pts[:, 1] + self.c)


@dataclass
class AutoCalibrationResult:
    calibration: Calibration | None
    score: float
    lines: list[ImageLine] = field(default_factory=list)
    message: str = ""
    mask: np.ndarray | None = None

    @property
    def success(self) -> bool:
        return self.calibration is not None


def line_mask(image: np.ndarray) -> np.ndarray:
    """Binary mask of thin bright low-saturation structures (court lines)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    k = max(9, int(round(image.shape[1] / 60)) | 1)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    v_thr = max(120, int(np.percentile(val, 80)))
    mask = (tophat > 30) & (sat < 90) & (val > v_thr * 0.8)
    mask = mask.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return mask


def _fit_line(points: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    w = weights / weights.sum()
    mean = (points * w[:, None]).sum(axis=0)
    d = points - mean
    cov = (d * w[:, None]).T @ d
    evals, evecs = np.linalg.eigh(cov)
    normal = evecs[:, 0]  # smallest eigenvalue -> normal direction
    a, b = normal
    c = -(a * mean[0] + b * mean[1])
    return float(a), float(b), float(c)


def extract_lines(mask: np.ndarray) -> list[ImageLine]:
    h, w = mask.shape
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=40, minLineLength=int(0.05 * w), maxLineGap=12)
    if segs is None:
        return []
    segs = np.asarray(segs).reshape(-1, 4).astype(float)  # OpenCV 4 gives (N,1,4), OpenCV 5 (N,4)
    groups: list[dict] = []
    for x1, y1, x2, y2 in segs:
        length = float(np.hypot(x2 - x1, y2 - y1))
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        family = "horizontal" if (ang < 20 or ang > 160) else "side"
        # Parameterise for clustering.
        if family == "horizontal":
            slope = (y2 - y1) / (x2 - x1 + 1e-9)
            key = y1 + slope * (w / 2 - x1)
            angle = np.arctan(slope)
        else:
            inv = (x2 - x1) / (y2 - y1 + 1e-9)
            key = x1 + inv * (h / 2 - y1)
            angle = np.arctan(inv)
        for g in groups:
            if g["family"] == family and abs(g["key"] - key) < 0.012 * w + 3 and abs(g["angle"] - angle) < 0.06:
                g["pts"].extend([(x1, y1), (x2, y2)])
                g["wts"].extend([length, length])
                g["support"] += length
                tot = g["support"]
                g["key"] = g["key"] + (key - g["key"]) * length / tot
                g["angle"] = g["angle"] + (angle - g["angle"]) * length / tot
                break
        else:
            groups.append(
                {
                    "family": family,
                    "key": key,
                    "angle": angle,
                    "pts": [(x1, y1), (x2, y2)],
                    "wts": [length, length],
                    "support": length,
                }
            )
    lines = []
    for g in groups:
        a, b, c = _fit_line(np.array(g["pts"]), np.array(g["wts"]))
        lines.append(ImageLine(a, b, c, g["support"], g["family"], g["key"]))
    return lines


def _quick_score(H_court_to_img: np.ndarray, samples: np.ndarray, dist: np.ndarray, size: tuple[int, int]) -> float:
    """Cheap per-sample score used to shortlist hypotheses."""
    w, h = size
    uv = project(H_court_to_img, samples)
    ok = np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
    n_in = int(ok.sum())
    if n_in < 0.25 * len(samples):
        return -1.0
    ui = uv[ok].astype(int)
    hits = float(np.clip(1.0 - dist[ui[:, 1], ui[:, 0]] / 4.0, 0, 1).sum())
    return hits / len(samples)


def _score(
    H_court_to_img: np.ndarray, samples: np.ndarray, dist: np.ndarray, size: tuple[int, int]
) -> tuple[float, float]:
    w, h = size
    uv = project(H_court_to_img, samples)
    ok = np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
    n_in = int(ok.sum())
    if n_in < 0.25 * len(samples):
        return -1.0, 0.0
    # Count distinct pixels rather than samples: a degenerate homography that squashes the whole
    # court onto one painted line would otherwise score perfectly.
    ui = np.unique(uv[ok].astype(int) // 2, axis=0) * 2
    d = dist[ui[:, 1], ui[:, 0]]
    hits = float(np.clip(1.0 - d / 4.0, 0, 1).sum())
    precision = hits / len(ui)
    return hits - 0.5 * (len(ui) - hits), precision


def _interior_points() -> np.ndarray:
    """Court-floor points at least 0.25 m away from every painted line."""
    xs = np.arange(0.1, 6.1, 0.2)
    ys = np.arange(0.1, 13.4, 0.2)
    g = np.array([[x, y] for x in xs for y in ys])
    lx = np.array(list(VERTICAL_LINES.values()))
    ly = np.array(list(HORIZONTAL_LINES.values()))
    dx = np.min(np.abs(g[:, :1] - lx[None, :]), axis=1)
    dy = np.min(np.abs(g[:, 1:] - ly[None, :]), axis=1)
    return g[(dx > 0.25) & (dy > 0.25) & (np.abs(g[:, 1] - 6.7) > 0.3)]


_INTERIOR = _interior_points()


def _interior_on_mask(H_court_to_img: np.ndarray, dist: np.ndarray, size: tuple[int, int]) -> float | None:
    w, h = size
    uv = project(H_court_to_img, _INTERIOR)
    ok = np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
    if ok.sum() < 20:
        return None
    ui = np.unique(uv[ok].astype(int), axis=0)
    return float(np.mean(dist[ui[:, 1], ui[:, 0]] < 1.5))


def _is_valid_quad(pts: np.ndarray) -> bool:
    if not np.all(np.isfinite(pts)):
        return False
    return bool(cv2.isContourConvex(pts.astype(np.float32).reshape(-1, 1, 2)))


def auto_calibrate(
    image: np.ndarray, *, top_h: int = 8, top_s: int = 6, min_precision: float = 0.55, max_interior_hits: float = 0.15
) -> AutoCalibrationResult:
    """Detect the court in a BGR frame and return a validated calibration suggestion."""
    H0, W0 = image.shape[:2]
    scale = min(1.0, _WORK_WIDTH / W0)
    work = (
        cv2.resize(image, (round(W0 * scale), round(H0 * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else image
    )
    h, w = work.shape[:2]
    mask = line_mask(work)
    lines = extract_lines(mask)
    horiz = sorted([ln for ln in lines if ln.family == "horizontal"], key=lambda ln: -ln.support)[:top_h]
    side = sorted([ln for ln in lines if ln.family == "side"], key=lambda ln: -ln.support)[:top_s]
    if len(horiz) < 2 or len(side) < 2:
        return AutoCalibrationResult(None, 0.0, lines, "not enough court lines detected", mask)
    horiz.sort(key=lambda ln: ln.key)  # top of image (far) -> bottom (near)
    side.sort(key=lambda ln: ln.key)  # left -> right

    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    samples = sample_court_lines(0.10)
    coarse = sample_court_lines(0.30)
    model_h = [HORIZONTAL_LINES[k] for k in _HORIZ_MODEL]
    model_s = [VERTICAL_LINES[k] for k in _SIDE_MODEL]

    shortlist: list[tuple[float, np.ndarray]] = []
    for hi, hj in combinations(range(len(horiz)), 2):  # hi above hj in the image -> hi is farther
        for si, sj in combinations(range(len(side)), 2):
            corners = [
                horiz[hi].intersect(side[si]),
                horiz[hi].intersect(side[sj]),
                horiz[hj].intersect(side[sj]),
                horiz[hj].intersect(side[si]),
            ]
            if any(c is None for c in corners):
                continue
            img_q = np.array(corners)
            if not _is_valid_quad(img_q) or cv2.contourArea(img_q.astype(np.float32)) < 0.01 * w * h:
                continue
            # Allow intersections somewhat outside the frame (partially visible court).
            if np.any(np.abs(img_q[:, 0] - w / 2) > 2.5 * w) or np.any(np.abs(img_q[:, 1] - h / 2) > 2.5 * h):
                continue
            for ma, mb in combinations(range(len(model_h)), 2):  # ma nearer, mb farther
                y_far, y_near = model_h[mb], model_h[ma]
                for mc, md in combinations(range(len(model_s)), 2):
                    x_l, x_r = model_s[mc], model_s[md]
                    court_q = np.array([[x_l, y_far], [x_r, y_far], [x_r, y_near], [x_l, y_near]], np.float32)
                    Hc2i = cv2.getPerspectiveTransform(court_q, img_q.astype(np.float32))
                    q = _quick_score(Hc2i, coarse, dist, (w, h))
                    if q > 0.2:
                        shortlist.append((q, Hc2i))
    shortlist.sort(key=lambda item: -item[0])
    best = (-np.inf, 0.0, None)
    for _q, Hc2i in shortlist[:300]:
        s, prec = _score(Hc2i, samples, dist, (w, h))
        if s > best[0]:
            best = (s, prec, Hc2i)
    score, precision, Hc2i = best
    if Hc2i is None:
        return AutoCalibrationResult(None, 0.0, lines, "no consistent court hypothesis found", mask)
    if precision < min_precision:
        return AutoCalibrationResult(
            None, precision, lines, f"best court hypothesis only matched {precision:.0%} of projected lines", mask
        )
    # Negative evidence: the floor *between* painted lines must not look like line pixels. Cluttered
    # scenes (signage, white walls, vehicles) make any hypothesis "hit" lines by chance.
    interior = _interior_on_mask(Hc2i, dist, (w, h))
    if interior is None or interior > max_interior_hits:
        return AutoCalibrationResult(
            None,
            precision,
            lines,
            "line-like structure everywhere (not a clean court view); calibrate manually",
            mask,
        )

    # Refinement: snap detected lines to projected model lines, use all matched intersections.
    def match(lines_f: list[ImageLine], model: dict[str, float], axis: str) -> dict[str, ImageLine]:
        """One-to-one assignment of detected lines to projected model lines (Hungarian)."""
        names = list(model)
        cost = np.full((len(lines_f), len(names)), 1e6)
        for i, ln in enumerate(lines_f):
            for j, name in enumerate(names):
                val = model[name]
                seg = (
                    np.array([[val, t] for t in np.linspace(0, 13.4, 30)])
                    if axis == "x"
                    else np.array([[t, val] for t in np.linspace(0, 6.1, 30)])
                )
                uv = project(Hc2i, seg)
                inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
                if inside.sum() >= 5:
                    cost[i, j] = float(np.median(ln.distance(uv[inside])))
        rows, cols = linear_sum_assignment(cost)
        return {names[j]: lines_f[i] for i, j in zip(rows, cols, strict=True) if cost[i, j] < 6.0}

    vert_model = {k: v for k, v in VERTICAL_LINES.items() if k != "center"}
    mh = match(horiz, HORIZONTAL_LINES, "y")
    ms = match(side, vert_model, "x")
    img_pts, ids = [], []
    for hn, hl in mh.items():
        for sn, sl in ms.items():
            p = hl.intersect(sl)
            if p is not None and -0.1 * w <= p[0] <= 1.1 * w and -0.1 * h <= p[1] <= 1.1 * h:
                img_pts.append(p / scale)
                ids.append(f"{hn}__{sn}")
    S = np.diag([scale, scale, 1.0])  # original px -> working px
    cal = None
    reason = ""
    try:
        if len(img_pts) < 4:
            raise CalibrationError("too few matched intersections")
        cal = fit_homography(np.array(img_pts), point_ids=ids, method="auto", image_size=(W0, H0))
        refined_score, refined_precision = _score(S @ np.linalg.inv(cal.H), samples, dist, (w, h))
        if refined_precision < precision - 0.03 or refined_score < 0.9 * score:  # refinement made it worse
            reason = "refinement scored worse than the initial hypothesis"
            cal = None
    except CalibrationError as exc:
        reason = str(exc)
    if cal is None:
        # Fall back to the hypothesis itself, expressed as landmarks the user can edit.
        H_court_to_orig = np.linalg.inv(S) @ Hc2i
        corner_ids = [
            "near_baseline__left_doubles",
            "near_baseline__right_doubles",
            "far_baseline__right_doubles",
            "far_baseline__left_doubles",
            "near_short_service__left_doubles",
            "near_short_service__right_doubles",
            "far_short_service__left_doubles",
            "far_short_service__right_doubles",
        ]
        court_pts = np.array([[keypoint_by_id(k).x, keypoint_by_id(k).y] for k in corner_ids])
        pts = project(H_court_to_orig, court_pts)
        keep = (pts[:, 0] > -0.2 * W0) & (pts[:, 0] < 1.2 * W0) & (pts[:, 1] > -0.2 * H0) & (pts[:, 1] < 1.2 * H0)
        try:
            cal = fit_homography(
                pts[keep],
                point_ids=[k for k, ok in zip(corner_ids, keep, strict=True) if ok],
                method="auto",
                image_size=(W0, H0),
            )
        except CalibrationError as exc:
            return AutoCalibrationResult(None, precision, lines, f"refinement failed: {reason or exc}", mask)
    cal.warnings.insert(0, f"automatic detection matched {precision:.0%} of court line pixels; verify before use")
    return AutoCalibrationResult(cal, precision, lines, "ok", mask)
