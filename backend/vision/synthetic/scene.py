"""Synthetic badminton footage with exact ground truth.

This module renders a stylised indoor court and two humanoid players through a pinhole camera.
Because every pixel comes from known 3-D geometry, it provides exact ground truth for the
homography, player positions, bounding boxes, 2-D landmarks and stroke timings.

It exists to (1) test the pipeline deterministically, (2) power the clearly-labelled demo mode,
and (3) exercise the evaluation code. **Nothing produced here is real athletic data** and the
rendered players are not realistic enough for real detectors (YOLO, MediaPipe) to be judged on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from vision.calibration.homography import calibration_from_camera
from vision.geometry.court import (
    CENTER_X,
    COURT_LENGTH,
    COURT_WIDTH,
    NET_Y,
    from_half_coords,
    sample_court_lines,
)
from vision.pose.landmarks import NUM_LANDMARKS

NET_HEIGHT = 1.55


# --------------------------------------------------------------------------------- camera


@dataclass
class Camera:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    width: int
    height: int

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project (N, 3) world points. Returns pixel coords (N, 2) and camera depth (N,)."""
        X = np.asarray(X, float).reshape(-1, 3)
        cam = X @ self.R.T + self.t.reshape(1, 3)
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = (cam @ self.K.T)[:, :2] / z[:, None]
        return uv, z

    @property
    def homography(self) -> np.ndarray:
        """Exact image → court homography of the floor plane."""
        return calibration_from_camera(self.K, self.R, self.t)


def look_at_camera(position, target, width: int, height: int, fill: float = 0.92) -> Camera:
    C = np.asarray(position, float)
    T = np.asarray(target, float)
    f = T - C
    f /= np.linalg.norm(f)
    up = np.array([0.0, 0.0, 1.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.vstack([r, d, f])
    t = -R @ C
    # Pick the focal length so the whole doubles court (plus the net posts) fills the frame.
    corners = np.array(
        [
            [0, 0, 0],
            [COURT_WIDTH, 0, 0],
            [0, COURT_LENGTH, 0],
            [COURT_WIDTH, COURT_LENGTH, 0],
            [0, NET_Y, NET_HEIGHT],
            [COURT_WIDTH, NET_Y, NET_HEIGHT],
            [CENTER_X, 0, 2.3],
        ],
        float,
    )
    cam = corners @ R.T + t
    nx = np.abs(cam[:, 0] / cam[:, 2]).max()
    ny = np.abs(cam[:, 1] / cam[:, 2]).max()
    focal = fill * min((width / 2) / nx, (height / 2) / ny)
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])
    return Camera(K=K, R=R, t=t, width=width, height=height)


def default_camera(width: int = 1280, height: int = 720, view: str = "baseline") -> Camera:
    """A stationary camera on a tripod behind the near baseline (the supported setup)."""
    if view == "baseline":
        return look_at_camera((CENTER_X, -5.0, 6.5), (CENTER_X, 6.0, 0.0), width, height)
    if view == "corner":
        return look_at_camera((-2.0, -4.0, 6.0), (CENTER_X, 6.5, 0.0), width, height)
    raise ValueError(f"unknown view {view}")


# ------------------------------------------------------------------------------ movement


def _min_jerk(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


@dataclass
class StrokeTruth:
    t: float
    player: int
    shot_type: str  # overhead | drive | net | lift
    stroke_side: str  # forehand | backhand
    zone_hint: str


@dataclass
class PlayerTimeline:
    """Dense per-frame state for one player."""

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    side: int
    strokes: list[StrokeTruth] = field(default_factory=list)


_TARGETS = [
    # (name, lateral, depth, shot_type, weight)
    ("front_left", 0.55, 1.05, "net", 1.0),
    ("front_right", 4.6, 1.05, "net", 1.0),
    ("mid_left", 0.35, 3.3, "drive", 0.8),
    ("mid_right", 4.85, 3.3, "drive", 0.8),
    ("rear_left", 0.65, 5.9, "overhead", 1.2),
    ("rear_right", 4.55, 5.9, "overhead", 1.2),
    ("rear_center", 2.6, 6.0, "overhead", 0.6),
    ("front_center", 2.6, 0.9, "lift", 0.4),
]


def simulate_player(duration: float, fps: float, side: int, seed: int, intensity: float = 1.0) -> PlayerTimeline:
    """Rally-like movement: base → shot location (min-jerk) → stroke → recover to base.

    Positions are the midpoint between the feet in court metres. ``intensity`` scales movement
    speed so sessions can differ (used to make the comparison demo meaningful).
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration * fps))
    t = np.arange(n) / fps
    base_lat, base_depth = 2.59, 3.2
    lat = np.full(n, base_lat)
    dep = np.full(n, base_depth)
    strokes: list[StrokeTruth] = []

    names = [tg[0] for tg in _TARGETS]
    weights = np.array([tg[4] for tg in _TARGETS])
    weights = weights / weights.sum()

    cur_t = 0.6
    cur = np.array([base_lat, base_depth])

    def move(to: np.ndarray, start: float, dur: float) -> None:
        nonlocal cur
        i0, i1 = int(start * fps), min(n, int((start + dur) * fps) + 1)
        if i0 >= n:
            return
        s = _min_jerk((t[i0:i1] - start) / dur)
        lat[i0:i1] = cur[0] + s * (to[0] - cur[0])
        dep[i0:i1] = cur[1] + s * (to[1] - cur[1])
        lat[i1:] = to[0]
        dep[i1:] = to[1]
        cur = to.copy()

    while cur_t < duration - 2.0:
        rally_len = int(rng.integers(5, 11))
        for _ in range(rally_len):
            k = rng.choice(len(names), p=weights)
            name, tl, td, shot, _w = _TARGETS[k]
            target = np.array([tl + rng.normal(0, 0.2), td + rng.normal(0, 0.15)])
            dist = float(np.linalg.norm(target - cur))
            dur = (0.45 + dist / 3.2) / intensity
            move(target, cur_t, dur)
            contact = cur_t + dur + 0.05
            stroke_side = "forehand" if tl > 2.0 else "backhand"
            if shot == "overhead":
                stroke_side = "forehand"
            strokes.append(StrokeTruth(round(contact, 3), side, shot, stroke_side, name))
            hold = 0.3
            ret_target = np.array([base_lat + rng.normal(0, 0.25), base_depth + rng.normal(0, 0.25)])
            rdur = (0.45 + float(np.linalg.norm(ret_target - target)) / 3.0) / intensity
            move(ret_target, contact + hold, rdur)
            cur_t = contact + hold + rdur + float(rng.uniform(0.35, 0.9))
            if cur_t > duration - 2.0:
                break
        # Rest between rallies: slow walk around the base.
        rest = float(rng.uniform(2.0, 3.5))
        wander = np.array([base_lat + rng.normal(0, 0.5), base_depth + rng.normal(0.4, 0.3)])
        move(wander, cur_t, rest * 0.6)
        cur_t += rest

    lat += rng.normal(0, 0.004, n)  # tiny sway so stationary periods are not perfectly flat
    dep += rng.normal(0, 0.004, n)
    x, y = from_half_coords(lat, dep, side)
    return PlayerTimeline(t=t, x=x, y=y, side=side, strokes=strokes)


# ----------------------------------------------------------------------------- body model

_SWING_KEYS = {
    # (lateral, forward, height) for the racket-hand wrist at backswing, contact, follow-through.
    ("overhead", "forehand"): [(0.38, -0.20, 1.80), (0.22, 0.25, 2.28), (-0.25, 0.45, 1.05)],
    ("drive", "forehand"): [(0.70, -0.30, 1.35), (0.60, 0.45, 1.35), (-0.25, 0.40, 1.30)],
    ("drive", "backhand"): [(-0.45, -0.10, 1.40), (-0.60, 0.45, 1.35), (0.45, 0.30, 1.45)],
    ("net", "forehand"): [(0.45, 0.10, 0.80), (0.35, 0.75, 0.75), (0.25, 0.80, 1.15)],
    ("net", "backhand"): [(-0.30, 0.10, 0.80), (-0.35, 0.75, 0.75), (-0.10, 0.80, 1.15)],
    ("lift", "forehand"): [(0.45, -0.10, 0.70), (0.35, 0.60, 0.70), (0.30, 0.55, 1.75)],
    ("lift", "backhand"): [(-0.30, -0.10, 0.70), (-0.35, 0.60, 0.70), (-0.20, 0.55, 1.75)],
}
_REST_WRIST = (0.32, 0.42, 1.18)


def _smoothstep(a, b, s):
    s = np.clip(s, 0, 1)
    s = s * s * (3 - 2 * s)
    return np.asarray(a, float) + s * (np.asarray(b, float) - np.asarray(a, float))


def racket_wrist(dt: float, shot: str, stroke_side: str) -> np.ndarray:
    """Wrist position (local frame) at time ``dt`` relative to the contact instant."""
    keys = _SWING_KEYS.get((shot, stroke_side)) or _SWING_KEYS[(shot, "forehand")]
    back, contact, follow = keys
    if dt < -0.40 or dt > 0.45:
        return np.asarray(_REST_WRIST, float)
    if dt < -0.15:
        return _smoothstep(_REST_WRIST, back, (dt + 0.40) / 0.25)
    if dt < 0.0:
        return _smoothstep(back, contact, (dt + 0.15) / 0.15)
    if dt < 0.20:
        return _smoothstep(contact, follow, dt / 0.20)
    return _smoothstep(follow, _REST_WRIST, (dt - 0.20) / 0.25)


def body_landmarks_3d(
    x: float, y: float, facing: np.ndarray, crouch: float, stance: float, lunge: float, wrist_local: np.ndarray
) -> np.ndarray:
    """33 MediaPipe-ordered 3-D landmarks (world metres) for a right-handed player."""
    P = np.array([x, y, 0.0])
    F = np.array([facing[0], facing[1], 0.0])
    F /= np.linalg.norm(F)
    Rv = np.array([F[1], -F[0], 0.0])
    U = np.array([0.0, 0.0, 1.0])

    def pt(lat, fwd, h):
        return P + lat * Rv + fwd * F + h * U

    c = crouch
    L = np.zeros((NUM_LANDMARKS, 3))
    head_h = 1.62 - 0.28 * c
    L[0] = pt(0, 0.10, head_h)
    L[1], L[2], L[3] = pt(-0.02, 0.09, head_h + 0.04), pt(-0.035, 0.09, head_h + 0.04), pt(-0.05, 0.08, head_h + 0.04)
    L[4], L[5], L[6] = pt(0.02, 0.09, head_h + 0.04), pt(0.035, 0.09, head_h + 0.04), pt(0.05, 0.08, head_h + 0.04)
    L[7], L[8] = pt(-0.075, 0.0, head_h + 0.02), pt(0.075, 0.0, head_h + 0.02)
    L[9], L[10] = pt(-0.025, 0.09, head_h - 0.04), pt(0.025, 0.09, head_h - 0.04)
    sh_h = 1.43 - 0.27 * c
    L[11], L[12] = pt(-0.19, 0.05 * c, sh_h), pt(0.19, 0.05 * c, sh_h)
    hip_h = 0.95 - 0.22 * c
    L[23], L[24] = pt(-0.11, 0, hip_h), pt(0.11, 0, hip_h)
    # Right leg leads in a lunge.
    L[27] = pt(-stance, 0.0, 0.09)
    L[28] = pt(stance, lunge, 0.09)
    L[25] = pt(-stance * 0.85, 0.10 * c, 0.50 - 0.12 * c)
    L[26] = pt(stance * 0.85, 0.10 * c + lunge * 0.55, 0.50 - 0.12 * c)
    for ank, heel, toe in ((27, 29, 31), (28, 30, 32)):
        L[heel] = L[ank] + (-0.06 * F) + np.array([0, 0, -0.06])
        L[toe] = L[ank] + (0.17 * F) + np.array([0, 0, -0.07])
    # Non-racket (left) arm at rest; racket (right) arm follows the swing keyframes.
    L[13] = pt(-0.30, 0.15, sh_h - 0.28)
    L[15] = pt(-0.26, 0.38, sh_h - 0.25)
    wr = pt(*wrist_local)
    wr[2] -= 0.27 * c * (1.0 if wrist_local[2] < 1.6 else 0.3)
    L[16] = wr
    sh = L[12]
    L[14] = sh + 0.52 * (wr - sh) + 0.10 * Rv - np.array([0, 0, 0.06])
    for el, w, pinky, index, thumb in ((13, 15, 17, 19, 21), (14, 16, 18, 20, 22)):
        d = L[w] - L[el]
        d /= np.linalg.norm(d) + 1e-9
        L[pinky] = L[w] + 0.07 * d - 0.02 * Rv
        L[index] = L[w] + 0.08 * d + 0.01 * Rv
        L[thumb] = L[w] + 0.05 * d + 0.03 * Rv
    return L


@dataclass
class PlayerFrame:
    landmarks3d: np.ndarray
    racket_tip: np.ndarray


def player_state(tl: PlayerTimeline, i: int, fps: float) -> PlayerFrame:
    x, y = tl.x[i], tl.y[i]
    # Face the net, turning partly towards the direction of travel.
    facing_net = np.array([0.0, 1.0]) if tl.side == 0 else np.array([0.0, -1.0])
    j0, j1 = max(0, i - 2), min(len(tl.t) - 1, i + 2)
    vel = np.array([tl.x[j1] - tl.x[j0], tl.y[j1] - tl.y[j0]]) * fps / max(1, j1 - j0)
    speed = float(np.linalg.norm(vel))
    facing = facing_net.copy()
    if speed > 0.5:
        facing = facing_net + 0.35 * vel / speed
    t = tl.t[i]
    stroke = min(tl.strokes, key=lambda s: abs(s.t - t)) if tl.strokes else None
    dt = (t - stroke.t) if stroke else 99.0
    shot = stroke.shot_type if stroke else "drive"
    side = stroke.stroke_side if stroke else "forehand"
    wrist = racket_wrist(dt, shot, side) if abs(dt) < 0.5 else np.asarray(_REST_WRIST, float)
    crouch = 0.35 + 0.25 * min(speed / 3.0, 1.0)
    stance = 0.22 + 0.06 * min(speed / 3.0, 1.0)
    lunge = 0.0
    if stroke and shot in ("net", "lift") and -0.25 < dt < 0.45:
        lunge = 0.55 * float(np.clip(1 - abs(dt - 0.05) / 0.4, 0, 1))
        crouch = max(crouch, 0.6)
    L = body_landmarks_3d(x, y, facing, crouch, stance, lunge, wrist)
    d = L[16] - L[14]
    racket_tip = L[16] + 0.62 * d / (np.linalg.norm(d) + 1e-9)
    return PlayerFrame(landmarks3d=L, racket_tip=racket_tip)


# -------------------------------------------------------------------------------- render

_COLORS = {
    "surround": (58, 64, 70),
    "court": (72, 110, 48),
    "line": (238, 240, 240),
    "net": (40, 40, 40),
    "skin": (150, 180, 220),
    "shorts": (40, 40, 40),
    "shoe": (230, 230, 230),
}
_SHIRTS = [(200, 160, 30), (40, 40, 200)]  # BGR: player 0 teal-ish, player 1 red


def _draw_limb(img, cam: Camera, a, b, width_m, color):
    uv, z = cam.project(np.vstack([a, b]))
    if np.any(z <= 0.1):
        return
    th = max(1, int(round(cam.K[0, 0] * width_m / float(np.mean(z)))))
    cv2.line(img, tuple(np.round(uv[0]).astype(int)), tuple(np.round(uv[1]).astype(int)), color, th, cv2.LINE_AA)


def _draw_player(img, cam: Camera, pf: PlayerFrame, shirt) -> None:
    L = pf.landmarks3d
    # Shadow on the floor.
    foot_mid = (L[27] + L[28]) / 2
    shadow = np.array(
        [[foot_mid[0] + 0.35 * np.cos(a), foot_mid[1] + 0.22 * np.sin(a), 0] for a in np.linspace(0, 2 * np.pi, 20)]
    )
    uv, _ = cam.project(shadow)
    overlay = img.copy()
    cv2.fillPoly(overlay, [np.round(uv).astype(np.int32)], (30, 40, 25))
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, dst=img)
    # Legs.
    for hip, knee, ank, toe in ((23, 25, 27, 31), (24, 26, 28, 32)):
        _draw_limb(img, cam, L[hip], L[knee], 0.15, _COLORS["shorts"])
        _draw_limb(img, cam, L[knee], L[ank], 0.11, _COLORS["skin"])
        _draw_limb(img, cam, L[ank], L[toe], 0.10, _COLORS["shoe"])
    # Torso.
    uv, z = cam.project(L[[11, 12, 24, 23]])
    if np.all(z > 0.1):
        cv2.fillConvexPoly(img, np.round(uv).astype(np.int32), shirt, cv2.LINE_AA)
    _draw_limb(img, cam, (L[11] + L[12]) / 2, (L[23] + L[24]) / 2, 0.30, shirt)
    # Arms and racket.
    for sh, el, wr in ((11, 13, 15), (12, 14, 16)):
        _draw_limb(img, cam, L[sh], L[el], 0.10, shirt)
        _draw_limb(img, cam, L[el], L[wr], 0.08, _COLORS["skin"])
    _draw_limb(img, cam, L[16], pf.racket_tip, 0.02, (20, 20, 20))
    uv, z = cam.project(pf.racket_tip[None])
    if z[0] > 0.1:
        r = max(2, int(cam.K[0, 0] * 0.11 / z[0]))
        cv2.ellipse(img, tuple(np.round(uv[0]).astype(int)), (r, int(r * 1.3)), 0, 0, 360, (20, 20, 20), 1, cv2.LINE_AA)
    # Head.
    uv, z = cam.project(((L[7] + L[8]) / 2)[None])
    if z[0] > 0.1:
        r = max(2, int(cam.K[0, 0] * 0.11 / z[0]))
        cv2.circle(img, tuple(np.round(uv[0]).astype(int)), r, _COLORS["skin"], -1, cv2.LINE_AA)
        cv2.circle(img, tuple(np.round(uv[0] - [0, r * 0.4]).astype(int)), int(r * 0.85), (30, 30, 40), -1, cv2.LINE_AA)


def render_background(cam: Camera) -> np.ndarray:
    img = np.empty((cam.height, cam.width, 3), np.uint8)
    img[:] = _COLORS["surround"]
    # Court surface slightly larger than the lines (run-off area).
    surf = np.array(
        [
            [-1.2, -1.5, 0],
            [COURT_WIDTH + 1.2, -1.5, 0],
            [COURT_WIDTH + 1.2, COURT_LENGTH + 1.5, 0],
            [-1.2, COURT_LENGTH + 1.5, 0],
        ]
    )
    uv, z = cam.project(surf)
    if np.all(z > 0):
        cv2.fillConvexPoly(img, np.round(uv).astype(np.int32), _COLORS["court"], cv2.LINE_AA)
    pts = sample_court_lines(0.02)
    uv, z = cam.project(np.hstack([pts, np.zeros((len(pts), 1))]))
    for (u, v), depth in zip(uv, z, strict=True):
        if depth <= 0:
            continue
        r = max(1, int(round(cam.K[0, 0] * 0.025 / depth)))
        cv2.circle(img, (int(round(u)), int(round(v))), r, _COLORS["line"], -1)
    return img


def _draw_net(img, cam: Camera) -> None:
    corners = np.array(
        [[0, NET_Y, 0.76], [COURT_WIDTH, NET_Y, 0.76], [COURT_WIDTH, NET_Y, NET_HEIGHT], [0, NET_Y, NET_HEIGHT]]
    )
    uv, z = cam.project(corners)
    if np.any(z <= 0):
        return
    overlay = img.copy()
    cv2.fillConvexPoly(overlay, np.round(uv).astype(np.int32), _COLORS["net"], cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, dst=img)
    cv2.line(
        img, tuple(np.round(uv[3]).astype(int)), tuple(np.round(uv[2]).astype(int)), (245, 245, 245), 2, cv2.LINE_AA
    )
    for x in (0.0, COURT_WIDTH):
        p, zz = cam.project(np.array([[x, NET_Y, 0], [x, NET_Y, NET_HEIGHT]]))
        if np.all(zz > 0):
            cv2.line(
                img, tuple(np.round(p[0]).astype(int)), tuple(np.round(p[1]).astype(int)), (30, 30, 30), 3, cv2.LINE_AA
            )


def player_bbox(cam: Camera, pf: PlayerFrame) -> np.ndarray:
    pts = np.vstack([pf.landmarks3d, pf.landmarks3d[0] + [0, 0, 0.14]])
    uv, _ = cam.project(pts)
    x0, y0 = uv.min(axis=0)
    x1, y1 = uv.max(axis=0)
    pw, ph = (x1 - x0) * 0.08, (y1 - y0) * 0.03
    return np.array([x0 - pw, y0 - ph, x1 + pw, y1 + ph])


@dataclass
class SyntheticScene:
    camera: Camera
    fps: float
    duration: float
    players: list[PlayerTimeline]

    @property
    def n_frames(self) -> int:
        return len(self.players[0].t)

    def frame_truth(self, i: int) -> list[dict]:
        out = []
        for pid, tl in enumerate(self.players):
            pf = player_state(tl, i, self.fps)
            uv, _ = self.camera.project(pf.landmarks3d)
            # Ground truth uses the same definition as the pipeline: midpoint of the two feet,
            # each foot being the mean of heel and toe (centre of the base of support).
            feet = pf.landmarks3d[[29, 30, 31, 32], :2].mean(axis=0)
            out.append(
                {
                    "player": pid,
                    "bbox": player_bbox(self.camera, pf).round(2).tolist(),
                    "landmarks": uv.round(2).tolist(),
                    "court_xy": [round(float(feet[0]), 4), round(float(feet[1]), 4)],
                    "body_xy": [round(float(tl.x[i]), 4), round(float(tl.y[i]), 4)],
                }
            )
        return out

    def render(self, i: int) -> np.ndarray:
        img = self._bg.copy()
        states = [player_state(tl, i, self.fps) for tl in self.players]
        far = [k for k, tl in enumerate(self.players) if tl.side == 1]
        near = [k for k, tl in enumerate(self.players) if tl.side == 0]
        for k in far:
            _draw_player(img, self.camera, states[k], _SHIRTS[k % 2])
        _draw_net(img, self.camera)
        for k in near:
            _draw_player(img, self.camera, states[k], _SHIRTS[k % 2])
        return img

    def __post_init__(self) -> None:
        self._bg = render_background(self.camera)


def make_scene(
    duration: float = 20.0,
    fps: float = 30.0,
    width: int = 1280,
    height: int = 720,
    seed: int = 7,
    intensity: float = 1.0,
    view: str = "baseline",
    opponent: bool = True,
) -> SyntheticScene:
    cam = default_camera(width, height, view)
    players = [simulate_player(duration, fps, side=0, seed=seed, intensity=intensity)]
    if opponent:
        players.append(simulate_player(duration, fps, side=1, seed=seed + 1000, intensity=0.9))
    return SyntheticScene(camera=cam, fps=fps, duration=duration, players=players)


def write_scene(scene: SyntheticScene, stem: str | Path) -> tuple[Path, Path]:
    """Render the scene to a video file plus a ``.truth.json`` ground-truth sidecar."""
    from vision.video.writer import open_writer

    writer, path, _codec = open_writer(stem, scene.fps, (scene.camera.width, scene.camera.height))
    frames = []
    try:
        for i in range(scene.n_frames):
            writer.write(scene.render(i))
            frames.append({"frame_index": i, "t": round(i / scene.fps, 4), "players": scene.frame_truth(i)})
    finally:
        writer.release()
    truth = {
        "synthetic": True,
        "description": "Synthetic BadmintonVision scene. Not real athletic data.",
        "fps": scene.fps,
        "duration_s": scene.duration,
        "width": scene.camera.width,
        "height": scene.camera.height,
        "homography_image_to_court": scene.camera.homography.tolist(),
        "target_player": 0,
        "frames": frames,
        "strokes": [s.__dict__ for tl in scene.players for s in tl.strokes],
    }
    truth_path = Path(str(stem) + ".truth.json")
    truth_path.write_text(json.dumps(truth))
    return path, truth_path


def court_keypoint_pixels(cam: Camera, ids: list[str]) -> np.ndarray:
    from vision.geometry.court import keypoint_by_id

    pts = np.array([[keypoint_by_id(k).x, keypoint_by_id(k).y, 0.0] for k in ids])
    uv, _ = cam.project(pts)
    return uv


def _graphic_frame(width: int, height: int, k: int) -> np.ndarray:
    """A broadcast-style interstitial (replay sting / logo) with no court in it."""
    img = np.zeros((height, width, 3), np.uint8)
    img[:] = (40 + (k * 3) % 40, 20, 60)
    cv2.putText(
        img,
        "REPLAY",
        (width // 4, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        width / 400,
        (240, 240, 240),
        max(2, width // 200),
        cv2.LINE_AA,
    )
    return img


def write_edited_scene(
    stem: str | Path,
    seeds: list[int],
    segment_s: float = 4.0,
    fps: float = 30.0,
    width: int = 640,
    height: int = 360,
    graphic_after: int | None = 1,
    graphic_s: float = 1.0,
) -> tuple[Path, Path, list[float]]:
    """Highlight-style edit: rally segments from the same camera cut together (players jump between
    positions at each cut), optionally with a non-court graphic spliced in after segment ``graphic_after``.

    Returns (video path, truth path, cut times). Truth frames of the graphic have no players.
    """
    from vision.video.writer import open_writer

    writer, path, _ = open_writer(stem, fps, (width, height))
    frames: list[dict] = []
    strokes: list[dict] = []
    cuts: list[float] = []
    k = 0
    cam = None
    try:
        rng = np.random.default_rng(sum(seeds))
        for si, seed in enumerate(seeds):
            # Highlights cut in mid-rally: start each segment 1.5–4 s into its own simulated rally, so
            # the players are somewhere else entirely when the picture cuts.
            skip = int(rng.uniform(1.5, 4.0) * fps)
            scene = make_scene(duration=segment_s + skip / fps, fps=fps, width=width, height=height, seed=seed)
            cam = scene.camera
            if k:
                cuts.append(round(k / fps, 4))
            t_off = k / fps - skip / fps
            for i in range(skip, scene.n_frames):
                writer.write(scene.render(i))
                frames.append(
                    {"frame_index": k, "t": round(k / fps, 4), "players": scene.frame_truth(i), "segment": si}
                )
                k += 1
            strokes += [
                {**s.__dict__, "t": round(s.t + t_off, 3)}
                for tl in scene.players
                for s in tl.strokes
                if s.t >= skip / fps
            ]
            if graphic_after is not None and si == graphic_after:
                cuts.append(round(k / fps, 4))
                for j in range(int(graphic_s * fps)):
                    writer.write(_graphic_frame(width, height, j))
                    frames.append({"frame_index": k, "t": round(k / fps, 4), "players": [], "segment": None})
                    k += 1
    finally:
        writer.release()
    truth = {
        "synthetic": True,
        "description": "Synthetic edited (highlight-style) video with cuts. Not real athletic data.",
        "fps": fps,
        "duration_s": k / fps,
        "width": width,
        "height": height,
        "homography_image_to_court": cam.homography.tolist(),
        "target_player": 0,
        "cuts": cuts,
        "frames": frames,
        "strokes": strokes,
    }
    truth_path = Path(str(stem) + ".truth.json")
    truth_path.write_text(json.dumps(truth))
    return path, truth_path, cuts
