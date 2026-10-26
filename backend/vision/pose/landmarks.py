"""MediaPipe Pose landmark indices and monocular joint-angle proxies.

All angles here are computed from 2-D image landmarks of a single camera. They are *proxies*
that track relative change within one session (e.g. "knees bent more during this lunge"),
not calibrated anatomical joint angles, and must never be used for medical judgements.
"""

from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 33

NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]
IDX = {name: i for i, name in enumerate(NAMES)}

NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32

# Skeleton edges for drawing (subset of MediaPipe's POSE_CONNECTIONS without the face mesh).
SKELETON = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (27, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
    (28, 32),
    (0, 11),
    (0, 12),
]

MIN_VIS = 0.5


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle ABC in degrees for arrays of 2-D points (..., 2)."""
    u = a - b
    w = c - b
    nu = np.linalg.norm(u, axis=-1)
    nw = np.linalg.norm(w, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(u * w, axis=-1) / (nu * nw)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _masked(values: np.ndarray, vis: np.ndarray, *idx: int, min_vis: float = MIN_VIS) -> np.ndarray:
    ok = np.all(vis[:, list(idx)] >= min_vis, axis=1)
    out = values.astype(float).copy()
    out[~ok] = np.nan
    return out


def joint_angle_series(xy: np.ndarray, vis: np.ndarray, side: str) -> dict[str, np.ndarray]:
    """Per-frame angle proxies for one body side.

    ``xy``: (T, 33, 2) image coordinates (pixels); ``vis``: (T, 33) visibility in [0, 1].
    NaN where the landmarks involved are not visible enough.
    """
    s = 0 if side == "left" else 1
    sh, el, wr = (L_SHOULDER, L_ELBOW, L_WRIST) if s == 0 else (R_SHOULDER, R_ELBOW, R_WRIST)
    hip, knee, ank = (L_HIP, L_KNEE, L_ANKLE) if s == 0 else (R_HIP, R_KNEE, R_ANKLE)
    return {
        f"{side}_elbow_angle": _masked(_angle(xy[:, sh], xy[:, el], xy[:, wr]), vis, sh, el, wr),
        f"{side}_knee_flexion": _masked(180.0 - _angle(xy[:, hip], xy[:, knee], xy[:, ank]), vis, hip, knee, ank),
        f"{side}_shoulder_angle": _masked(_angle(xy[:, hip], xy[:, sh], xy[:, el]), vis, hip, sh, el),
    }


def body_metrics(xy: np.ndarray, vis: np.ndarray) -> dict[str, np.ndarray]:
    """All per-frame pose proxies used by the dashboard and the stroke detector.

    * ``*_elbow_angle`` — interior elbow angle (180 = straight arm)
    * ``*_knee_flexion`` — 180 minus the hip-knee-ankle angle (0 = straight leg)
    * ``*_shoulder_angle`` — hip-shoulder-elbow angle (arm elevation relative to the trunk)
    * ``*_arm_elevation`` — wrist height above the shoulder in torso lengths (+ = above)
    * ``torso_lean_deg`` — angle of the hip→shoulder midline from image vertical
    * ``torso_orientation`` — shoulder width / torso length; small values mean the player is
      side-on to the camera, large values square-on (a 2-D proxy, not a 3-D rotation)
    * ``stance_width`` — ankle separation in torso lengths
    * ``hip_ankle_offset`` — horizontal hip-midpoint offset from the ankle midpoint, in torso lengths
    * ``torso_length_px`` — scale used for normalisation
    """
    out: dict[str, np.ndarray] = {}
    out.update(joint_angle_series(xy, vis, "left"))
    out.update(joint_angle_series(xy, vis, "right"))

    sh_mid = (xy[:, L_SHOULDER] + xy[:, R_SHOULDER]) / 2
    hip_mid = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
    ank_mid = (xy[:, L_ANKLE] + xy[:, R_ANKLE]) / 2
    torso = np.linalg.norm(sh_mid - hip_mid, axis=1)
    torso_ok = np.all(vis[:, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]] >= MIN_VIS, axis=1) & (torso > 1e-6)
    torso_len = np.where(torso_ok, torso, np.nan)
    out["torso_length_px"] = torso_len

    d = sh_mid - hip_mid
    out["torso_lean_deg"] = np.where(torso_ok, np.degrees(np.arctan2(d[:, 0], -d[:, 1])), np.nan)
    shoulder_w = np.linalg.norm(xy[:, L_SHOULDER] - xy[:, R_SHOULDER], axis=1)
    out["torso_orientation"] = shoulder_w / torso_len

    for side, (sh, wr) in {"left": (L_SHOULDER, L_WRIST), "right": (R_SHOULDER, R_WRIST)}.items():
        elev = (xy[:, sh, 1] - xy[:, wr, 1]) / torso_len  # image y grows downwards
        out[f"{side}_arm_elevation"] = _masked(elev, vis, sh, wr)

    ank_ok = np.all(vis[:, [L_ANKLE, R_ANKLE]] >= MIN_VIS, axis=1)
    stance = np.linalg.norm(xy[:, L_ANKLE] - xy[:, R_ANKLE], axis=1) / torso_len
    out["stance_width"] = np.where(ank_ok, stance, np.nan)
    out["hip_ankle_offset"] = np.where(ank_ok, (hip_mid[:, 0] - ank_mid[:, 0]) / torso_len, np.nan)
    return out
