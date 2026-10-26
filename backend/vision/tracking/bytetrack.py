"""ByteTrack multi-object tracker (Zhang et al., ECCV 2022), re-implemented from the paper.

Written from scratch (MIT) rather than vendored so that tracker state is an ordinary object
owned by one analysis job: two jobs never share a tracker, and ``reset()`` clears everything.

Core idea: associate high-confidence detections first, then try to rescue tracks with the
*low*-confidence detections (often partially occluded people) instead of discarding them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from vision.detection.base import Detection


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N, 4) and (M, 4) xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = np.asarray(a, float)[:, None, :]
    b = np.asarray(b, float)[None, :, :]
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


class KalmanBoxFilter:
    """Constant-velocity Kalman filter on (cx, cy, aspect, height), as in SORT/ByteTrack."""

    _std_pos = 1.0 / 20
    _std_vel = 1.0 / 160

    def __init__(self) -> None:
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0
        self.Hm = np.eye(4, 8)

    def initiate(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [
            2 * self._std_pos * h,
            2 * self._std_pos * h,
            1e-2,
            2 * self._std_pos * h,
            10 * self._std_vel * h,
            10 * self._std_vel * h,
            1e-5,
            10 * self._std_vel * h,
        ]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = [
            self._std_pos * h,
            self._std_pos * h,
            1e-2,
            self._std_pos * h,
            self._std_vel * h,
            self._std_vel * h,
            1e-5,
            self._std_vel * h,
        ]
        mean = self.F @ mean
        cov = self.F @ cov @ self.F.T + np.diag(np.square(std))
        return mean, cov

    def update(self, mean: np.ndarray, cov: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        R = np.diag(np.square([self._std_pos * h, self._std_pos * h, 1e-1, self._std_pos * h]))
        S = self.Hm @ cov @ self.Hm.T + R
        K = cov @ self.Hm.T @ np.linalg.inv(S)
        mean = mean + K @ (z - self.Hm @ mean)
        cov = (np.eye(8) - K @ self.Hm) @ cov
        return mean, cov


def xyxy_to_xyah(b: np.ndarray) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h])


def xyah_to_xyxy(s: np.ndarray) -> np.ndarray:
    cx, cy, a, h = s[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


@dataclass
class Track:
    track_id: int
    mean: np.ndarray
    cov: np.ndarray
    score: float
    start_frame: int
    last_frame: int
    hits: int = 1
    state: str = "tracked"  # tracked | lost | removed
    confirmed: bool = False
    last_det_bbox: np.ndarray = field(default_factory=lambda: np.zeros(4))

    @property
    def bbox(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean)


@dataclass
class TrackOutput:
    track_id: int
    bbox: np.ndarray
    score: float
    matched: bool  # True when a detection supported this track in this frame


@dataclass
class ByteTrackConfig:
    high_thresh: float = 0.5
    low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    match_iou: float = 0.2  # minimum IoU for the first association (ByteTrack: 1 - 0.8)
    second_match_iou: float = 0.5  # i.e. ByteTrack's 0.5 distance threshold for low-score dets
    unconfirmed_match_iou: float = 0.3
    track_buffer_s: float = 1.5  # how long a lost track can be revived
    min_hits_to_confirm: int = 2


class ByteTracker:
    def __init__(self, fps: float, config: ByteTrackConfig | None = None):
        self.cfg = config or ByteTrackConfig()
        self.fps = fps
        self.kf = KalmanBoxFilter()
        self.reset()

    def reset(self) -> None:
        self.tracks: list[Track] = []
        self.frame_id = 0
        self._next_id = 1
        self._shot_start = 1

    def new_shot(self) -> None:
        """Forget all tracks at a video cut, but keep ids unique across the whole video.

        Positions on either side of a cut are unrelated, so no box may be associated across it.
        """
        self.tracks = []
        self._shot_start = self.frame_id + 1

    @property
    def max_lost_frames(self) -> int:
        return max(1, int(round(self.cfg.track_buffer_s * self.fps)))

    def _associate(self, tracks: list[Track], dets: list[Detection], min_iou: float, fuse_score: bool):
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        iou = iou_matrix(np.array([t.bbox for t in tracks]), np.array([d.bbox for d in dets]))
        sim = iou * np.array([d.score for d in dets])[None, :] if fuse_score else iou
        cost = 1.0 - sim
        cost[iou < min_iou] = 1e6
        rows, cols = linear_sum_assignment(cost)
        matches = [(r, c) for r, c in zip(rows, cols, strict=True) if cost[r, c] < 1e5]
        mt = {r for r, _ in matches}
        md = {c for _, c in matches}
        return matches, [i for i in range(len(tracks)) if i not in mt], [j for j in range(len(dets)) if j not in md]

    def _apply(self, track: Track, det: Detection) -> None:
        track.mean, track.cov = self.kf.update(track.mean, track.cov, xyxy_to_xyah(det.bbox))
        track.score = det.score
        track.last_frame = self.frame_id
        track.hits += 1
        track.state = "tracked"
        track.last_det_bbox = det.bbox.copy()
        if track.hits >= self.cfg.min_hits_to_confirm:
            track.confirmed = True

    def update(self, detections: list[Detection]) -> list[TrackOutput]:
        self.frame_id += 1
        cfg = self.cfg
        high = [d for d in detections if d.score >= cfg.high_thresh]
        low = [d for d in detections if cfg.low_thresh <= d.score < cfg.high_thresh]

        for t in self.tracks:
            t.mean, t.cov = self.kf.predict(t.mean, t.cov)
            if t.state == "lost":
                t.mean[7] = 0.0  # do not let the height velocity drift while lost

        confirmed = [t for t in self.tracks if t.confirmed and t.state in ("tracked", "lost")]
        unconfirmed = [t for t in self.tracks if not t.confirmed and t.state == "tracked"]
        matched_ids: set[int] = set()

        # 1) high-score detections vs confirmed tracks (tracked + lost).
        m1, u_trk, u_det = self._associate(confirmed, high, cfg.match_iou, fuse_score=True)
        for r, c in m1:
            self._apply(confirmed[r], high[c])
            matched_ids.add(confirmed[r].track_id)
        # 2) low-score detections vs remaining *tracked* tracks.
        remaining = [confirmed[i] for i in u_trk if confirmed[i].state == "tracked"]
        m2, u_trk2, _ = self._associate(remaining, low, cfg.second_match_iou, fuse_score=False)
        for r, c in m2:
            self._apply(remaining[r], low[c])
            matched_ids.add(remaining[r].track_id)
        for i in u_trk2:
            remaining[i].state = "lost"
        # 3) unconfirmed tracks vs leftover high detections.
        left_high = [high[j] for j in u_det]
        m3, u_unc, u_det3 = self._associate(unconfirmed, left_high, cfg.unconfirmed_match_iou, fuse_score=True)
        for r, c in m3:
            self._apply(unconfirmed[r], left_high[c])
            matched_ids.add(unconfirmed[r].track_id)
        for i in u_unc:
            unconfirmed[i].state = "removed"
        # 4) new tracks.
        for j in u_det3:
            d = left_high[j]
            if d.score < cfg.new_track_thresh:
                continue
            mean, cov = self.kf.initiate(xyxy_to_xyah(d.bbox))
            tr = Track(self._next_id, mean, cov, d.score, self.frame_id, self.frame_id, last_det_bbox=d.bbox.copy())
            # Tracks born in the first frame (of the video or of a shot) are confirmed immediately, as in
            # the reference code.
            tr.confirmed = self.frame_id == self._shot_start
            self._next_id += 1
            self.tracks.append(tr)
            matched_ids.add(tr.track_id)
        # 5) expire lost tracks.
        for t in self.tracks:
            if t.state == "lost" and self.frame_id - t.last_frame > self.max_lost_frames:
                t.state = "removed"
        self.tracks = [t for t in self.tracks if t.state != "removed"]

        return [
            TrackOutput(t.track_id, t.last_det_bbox.copy(), t.score, True)
            for t in self.tracks
            if t.state == "tracked" and t.track_id in matched_ids and (t.confirmed or t.hits >= 1)
        ]
