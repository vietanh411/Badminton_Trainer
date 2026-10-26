"""Badminton court model in real-world metres (BWF Laws of Badminton, Appendix 1).

Coordinate system (used everywhere in the project):

* origin at the near-left doubles corner, as seen from a camera behind the near baseline
* ``x`` runs across the court, 0 → 6.10 m (left doubles sideline → right doubles sideline)
* ``y`` runs along the court, 0 → 13.40 m (near baseline → far baseline)
* the net is at ``y = 6.70``

Line positions are measured to the line centres; BWF lines are 40 mm wide, so the model has
±2 cm of inherent ambiguity, far below the accuracy of monocular foot localisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

COURT_LENGTH = 13.40
COURT_WIDTH = 6.10
SINGLES_WIDTH = 5.18
SIDE_ALLEY = (COURT_WIDTH - SINGLES_WIDTH) / 2  # 0.46
NET_Y = COURT_LENGTH / 2  # 6.70
SHORT_SERVICE_FROM_NET = 1.98
LONG_SERVICE_FROM_BACK = 0.76  # doubles long service line
CENTER_X = COURT_WIDTH / 2  # 3.05

# Lines parallel to the net (constant y).
HORIZONTAL_LINES: dict[str, float] = {
    "near_baseline": 0.0,
    "near_long_service": LONG_SERVICE_FROM_BACK,
    "near_short_service": NET_Y - SHORT_SERVICE_FROM_NET,
    "far_short_service": NET_Y + SHORT_SERVICE_FROM_NET,
    "far_long_service": COURT_LENGTH - LONG_SERVICE_FROM_BACK,
    "far_baseline": COURT_LENGTH,
}

# Lines perpendicular to the net (constant x). The centre line only exists between each
# short service line and its baseline.
VERTICAL_LINES: dict[str, float] = {
    "left_doubles": 0.0,
    "left_singles": SIDE_ALLEY,
    "center": CENTER_X,
    "right_singles": COURT_WIDTH - SIDE_ALLEY,
    "right_doubles": COURT_WIDTH,
}

_PRETTY = {
    "near_baseline": "near baseline",
    "near_long_service": "near doubles long-service line",
    "near_short_service": "near short-service line",
    "far_short_service": "far short-service line",
    "far_long_service": "far doubles long-service line",
    "far_baseline": "far baseline",
    "left_doubles": "left doubles sideline",
    "left_singles": "left singles sideline",
    "center": "centre line",
    "right_singles": "right singles sideline",
    "right_doubles": "right doubles sideline",
}


@dataclass(frozen=True)
class CourtKeypoint:
    id: str
    label: str
    x: float
    y: float


@dataclass(frozen=True)
class CourtLine:
    id: str
    label: str
    start: tuple[float, float]
    end: tuple[float, float]
    kind: str  # boundary | sideline | service | center | net


@lru_cache(maxsize=1)
def court_keypoints() -> tuple[CourtKeypoint, ...]:
    """Every painted line intersection (plus net-post bases) usable as a calibration landmark."""
    pts: list[CourtKeypoint] = []
    for h_id, y in HORIZONTAL_LINES.items():
        for v_id, x in VERTICAL_LINES.items():
            pts.append(CourtKeypoint(f"{h_id}__{v_id}", f"{_PRETTY[h_id]} × {_PRETTY[v_id]}", x, y))
    pts.append(CourtKeypoint("net_post_left", "left net-post base", 0.0, NET_Y))
    pts.append(CourtKeypoint("net_post_right", "right net-post base", COURT_WIDTH, NET_Y))
    return tuple(pts)


def keypoint_by_id(kp_id: str) -> CourtKeypoint:
    for kp in court_keypoints():
        if kp.id == kp_id:
            return kp
    raise KeyError(f"unknown court keypoint: {kp_id}")


@lru_cache(maxsize=1)
def court_lines() -> tuple[CourtLine, ...]:
    """Line segments of the full doubles court, in metres."""
    W, L = COURT_WIDTH, COURT_LENGTH
    lines = [
        CourtLine("near_baseline", "Near baseline", (0, 0), (W, 0), "boundary"),
        CourtLine("far_baseline", "Far baseline", (0, L), (W, L), "boundary"),
        CourtLine("left_doubles", "Left doubles sideline", (0, 0), (0, L), "boundary"),
        CourtLine("right_doubles", "Right doubles sideline", (W, 0), (W, L), "boundary"),
        CourtLine("left_singles", "Left singles sideline", (SIDE_ALLEY, 0), (SIDE_ALLEY, L), "sideline"),
        CourtLine("right_singles", "Right singles sideline", (W - SIDE_ALLEY, 0), (W - SIDE_ALLEY, L), "sideline"),
        CourtLine(
            "near_short_service",
            "Near short-service line",
            (0, HORIZONTAL_LINES["near_short_service"]),
            (W, HORIZONTAL_LINES["near_short_service"]),
            "service",
        ),
        CourtLine(
            "far_short_service",
            "Far short-service line",
            (0, HORIZONTAL_LINES["far_short_service"]),
            (W, HORIZONTAL_LINES["far_short_service"]),
            "service",
        ),
        CourtLine(
            "near_long_service",
            "Near doubles long-service line",
            (0, LONG_SERVICE_FROM_BACK),
            (W, LONG_SERVICE_FROM_BACK),
            "service",
        ),
        CourtLine(
            "far_long_service",
            "Far doubles long-service line",
            (0, L - LONG_SERVICE_FROM_BACK),
            (W, L - LONG_SERVICE_FROM_BACK),
            "service",
        ),
        CourtLine(
            "near_center",
            "Near centre line",
            (CENTER_X, 0),
            (CENTER_X, HORIZONTAL_LINES["near_short_service"]),
            "center",
        ),
        CourtLine(
            "far_center", "Far centre line", (CENTER_X, HORIZONTAL_LINES["far_short_service"]), (CENTER_X, L), "center"
        ),
        CourtLine("net", "Net", (0, NET_Y), (W, NET_Y), "net"),
    ]
    return tuple(lines)


def floor_lines() -> tuple[CourtLine, ...]:
    """Lines painted on the floor (the net is not on the floor plane)."""
    return tuple(line for line in court_lines() if line.kind != "net")


def sample_court_lines(spacing: float = 0.05, include_net: bool = False) -> np.ndarray:
    """Dense points along the painted lines, shape (N, 2), for rendering and scoring."""
    pts = []
    for line in court_lines():
        if line.kind == "net" and not include_net:
            continue
        a, b = np.asarray(line.start, float), np.asarray(line.end, float)
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / spacing)) + 1)
        t = np.linspace(0.0, 1.0, n)[:, None]
        pts.append(a + t * (b - a))
    return np.vstack(pts)


# ---------------------------------------------------------------------------- halves & zones


def side_of(y: float | np.ndarray) -> np.ndarray:
    """0 for the near half (y < net), 1 for the far half."""
    return (np.asarray(y) >= NET_Y).astype(int)


def to_half_coords(x: np.ndarray, y: np.ndarray, side: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert court coordinates into player-relative half coordinates.

    Returns ``(lateral, depth)`` where ``depth`` is the distance from the net (0 at the net,
    6.70 at the baseline) and ``lateral`` is measured from the player's own left singles
    sideline while facing the net (0 → 5.18 inside the singles court).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if side == 0:  # near half, player faces +y, so their left is x = 0
        return x - SIDE_ALLEY, NET_Y - y
    return (COURT_WIDTH - SIDE_ALLEY) - x, y - NET_Y


def from_half_coords(lateral: np.ndarray, depth: np.ndarray, side: int) -> tuple[np.ndarray, np.ndarray]:
    lateral = np.asarray(lateral, float)
    depth = np.asarray(depth, float)
    if side == 0:
        return lateral + SIDE_ALLEY, NET_Y - depth
    return (COURT_WIDTH - SIDE_ALLEY) - lateral, NET_Y + depth


@dataclass(frozen=True)
class ZoneConfig:
    """Zone grid over one singles half-court, in player-relative half coordinates.

    ``depth_edges`` are distances from the net. The defaults split the half at the short
    service line (front court) and halfway between it and the baseline (mid / rear).
    """

    depth_edges: tuple[float, ...] = (0.0, SHORT_SERVICE_FROM_NET, (SHORT_SERVICE_FROM_NET + NET_Y) / 2, NET_Y)
    depth_names: tuple[str, ...] = ("front", "mid", "rear")
    lateral_edges: tuple[float, ...] = (0.0, SINGLES_WIDTH / 3, 2 * SINGLES_WIDTH / 3, SINGLES_WIDTH)
    lateral_names: tuple[str, ...] = ("left", "center", "right")

    def __post_init__(self) -> None:
        if len(self.depth_edges) != len(self.depth_names) + 1:
            raise ValueError("depth_edges must have one more entry than depth_names")
        if len(self.lateral_edges) != len(self.lateral_names) + 1:
            raise ValueError("lateral_edges must have one more entry than lateral_names")
        if any(np.diff(self.depth_edges) <= 0) or any(np.diff(self.lateral_edges) <= 0):
            raise ValueError("zone edges must be strictly increasing")

    @property
    def zone_ids(self) -> list[str]:
        return [f"{d}_{lat}" for d in self.depth_names for lat in self.lateral_names]


def assign_zones(lateral: np.ndarray, depth: np.ndarray, cfg: ZoneConfig) -> np.ndarray:
    """Zone id for each half-coordinate point; ``None`` when outside the singles half or NaN.

    Points slightly outside the singles lines (e.g. a lunge past the sideline) are clipped into
    the nearest edge zone up to 0.5 m; anything further out is labelled ``"out"``.
    """
    lateral = np.asarray(lateral, float)
    depth = np.asarray(depth, float)
    out = np.empty(lateral.shape, dtype=object)
    out[:] = None
    tol = 0.5
    lo_l, hi_l = cfg.lateral_edges[0], cfg.lateral_edges[-1]
    lo_d, hi_d = cfg.depth_edges[0], cfg.depth_edges[-1]
    finite = np.isfinite(lateral) & np.isfinite(depth)
    inside = finite & (lateral >= lo_l - tol) & (lateral <= hi_l + tol) & (depth >= lo_d - 0.05) & (depth <= hi_d + tol)
    out[finite & ~inside] = "out"
    lc = np.clip(lateral, lo_l, hi_l - 1e-9)
    dc = np.clip(depth, lo_d, hi_d - 1e-9)
    li = np.clip(np.searchsorted(cfg.lateral_edges, lc, side="right") - 1, 0, len(cfg.lateral_names) - 1)
    di = np.clip(np.searchsorted(cfg.depth_edges, dc, side="right") - 1, 0, len(cfg.depth_names) - 1)
    for idx in np.flatnonzero(inside):
        out[idx] = f"{cfg.depth_names[di[idx]]}_{cfg.lateral_names[li[idx]]}"
    return out


def court_geometry_payload() -> dict:
    """JSON-serialisable court description for the frontend."""
    return {
        "length_m": COURT_LENGTH,
        "width_m": COURT_WIDTH,
        "singles_width_m": SINGLES_WIDTH,
        "net_y_m": NET_Y,
        "lines": [
            {"id": ln.id, "label": ln.label, "start": list(ln.start), "end": list(ln.end), "kind": ln.kind}
            for ln in court_lines()
        ],
        "keypoints": [{"id": k.id, "label": k.label, "x": k.x, "y": k.y} for k in court_keypoints()],
    }
