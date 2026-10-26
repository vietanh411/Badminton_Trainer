"""Ground-contact point of the player → court position.

The court homography is only valid on the floor, so the player's position must come from a
point that actually touches the floor. We use, in order of preference:

1. ``pose_feet`` — midpoint of both feet, each foot = mean of heel and toe (foot index), when
   both are visible: a proxy for the centre of the base of support.
2. ``pose_one_foot`` — a single visible foot (the other occluded).
3. ``bbox`` — bottom-centre of the detection box (shoes are the lowest part of a standing
   person). Less precise (lunges, wide stances, arms below feet in dives).

Hip or torso keypoints are never projected: they are ~1 m above the floor, so mapping them
through a floor homography displaces the position by metres depending on camera height.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vision.calibration.homography import project
from vision.pose.landmarks import L_ANKLE, L_FOOT, L_HEEL, R_ANKLE, R_FOOT, R_HEEL

SOURCE_CONF = {"pose_feet": 0.95, "pose_one_foot": 0.75, "pose_ankles": 0.65, "bbox": 0.45}


def _foot_point(xy: np.ndarray, vis: np.ndarray, heel: int, toe: int, ankle: int, min_vis: float):
    if vis[heel] >= min_vis and vis[toe] >= min_vis:
        return (xy[heel] + xy[toe]) / 2, (vis[heel] + vis[toe]) / 2, "foot"
    if vis[ankle] >= min_vis:
        return xy[ankle], vis[ankle], "ankle"
    return None, 0.0, None


def foot_contact_points(
    xy: np.ndarray, vis: np.ndarray, bbox: np.ndarray, det_score: np.ndarray, min_vis: float = 0.5
) -> pd.DataFrame:
    """Image-space ground-contact point per frame.

    ``xy``: (T, 33, 2) landmarks (NaN rows where pose failed), ``vis``: (T, 33),
    ``bbox``: (T, 4) target boxes (NaN where the target is missing), ``det_score``: (T,).
    """
    T = len(bbox)
    img = np.full((T, 2), np.nan)
    conf = np.zeros(T)
    source = np.array(["none"] * T, dtype=object)
    for i in range(T):
        b = bbox[i]
        if not np.all(np.isfinite(b)):
            continue
        bw, bh = b[2] - b[0], b[3] - b[1]
        foot_pts = []
        kinds = []
        if np.all(np.isfinite(xy[i])):
            for heel, toe, ank in ((L_HEEL, L_FOOT, L_ANKLE), (R_HEEL, R_FOOT, R_ANKLE)):
                p, v, kind = _foot_point(xy[i], vis[i], heel, toe, ank, min_vis)
                # Reject feet outside the (slightly expanded) detection box: pose drifted.
                if (
                    p is not None
                    and b[0] - 0.15 * bw <= p[0] <= b[2] + 0.15 * bw
                    and b[1] + 0.5 * bh <= p[1] <= b[3] + 0.1 * bh
                ):
                    foot_pts.append((p, v))
                    kinds.append(kind)
        score = float(det_score[i]) if np.isfinite(det_score[i]) else 0.5
        if len(foot_pts) == 2:
            img[i] = (foot_pts[0][0] + foot_pts[1][0]) / 2
            src = "pose_feet" if all(k == "foot" for k in kinds) else "pose_ankles"
            conf[i] = SOURCE_CONF[src] * (foot_pts[0][1] + foot_pts[1][1]) / 2
            source[i] = src
        elif len(foot_pts) == 1:
            img[i] = foot_pts[0][0]
            conf[i] = SOURCE_CONF["pose_one_foot"] * foot_pts[0][1]
            source[i] = "pose_one_foot"
        else:
            img[i] = [(b[0] + b[2]) / 2, b[3]]
            conf[i] = SOURCE_CONF["bbox"] * min(1.0, score / 0.6)
            source[i] = "bbox"
    return pd.DataFrame(
        {"img_x": img[:, 0], "img_y": img[:, 1], "position_source": source, "position_confidence": conf}
    )


def to_court(contact: pd.DataFrame, H: np.ndarray) -> pd.DataFrame:
    out = contact.copy()
    pts = project(H, contact[["img_x", "img_y"]].to_numpy(float))
    out["court_x_raw"] = pts[:, 0]
    out["court_y_raw"] = pts[:, 1]
    return out
