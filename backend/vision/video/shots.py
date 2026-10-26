"""Shot-boundary (cut) detection for edited footage such as highlight reels.

Highlights usually splice rallies filmed from the *same* camera, so the background hardly changes
across a cut and colour histograms barely move; only the players "teleport". The detector therefore
combines two cues computed on a tiny grey/HSV thumbnail of every sampled frame:

* **pixel-change spike** — the mean absolute difference to the previous sample is both large in
  absolute terms and several times larger than the recent typical difference (continuous motion
  ramps up gradually; a cut is a one-frame spike). Video-codec keyframe artefacts give small
  absolute changes and are ignored by the absolute floor.
* **histogram jump** — a large change of the hue/saturation histogram (a switch to a different
  camera, a replay graphic, a logo animation).

It is causal (only past samples are used), so the tracker can be reset at the moment of the cut
during the single decoding pass.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class ShotConfig:
    enabled: bool = True
    # Tuned on a 10.6 min broadcast highlight reel (38 same-view cuts found, 0 false at 15 fps); see docs.
    min_pixel_change: float = 0.009  # mean |Δ| of the grey thumbnail, as a fraction of 255
    spike_ratio: float = 3.0  # ... and ≥ this many times the median of the recent differences
    history: int = 6
    # Local cue: mean of the 4 most-changed cells of a 16×9 grid (a player vanishing / appearing).
    # Conservative on purpose: it caused no false cuts on the reference video.
    min_local_change: float = 0.03
    local_spike_ratio: float = 8.0
    hist_jump: float = 0.35  # half L1 distance between consecutive HSV histograms
    min_shot_s: float = 0.25  # a second cut within this time is merged into the first
    court_view_min_precision: float = 0.45  # share of projected court-line pixels on painted lines

    def to_dict(self) -> dict:
        return asdict(self)


class ShotDetector:
    """Feed every sampled frame in order; ``update`` returns True when a new shot starts."""

    def __init__(self, cfg: ShotConfig | None = None):
        self.cfg = cfg or ShotConfig()
        self.prev: tuple[np.ndarray, np.ndarray] | None = None
        self.recent: deque[float] = deque(maxlen=self.cfg.history)
        self.recent_local: deque[float] = deque(maxlen=self.cfg.history)
        self.last_cut_t = -1e9
        self.shot_id = 0
        self.scores: list[tuple[float, float, float, float]] = []  # (t, pixel, local, histogram change)

    @staticmethod
    def _features(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        small = cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA)
        grey = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).ravel()
        return grey, hist / max(float(hist.sum()), 1.0)

    def update(self, image: np.ndarray, t: float) -> bool:
        grey, hist = self._features(image)
        cut = False
        if self.prev is not None and self.cfg.enabled:
            diff = np.abs(grey - self.prev[0]) / 255.0
            pix = float(diff.mean())
            cells = cv2.resize(diff, (16, 9), interpolation=cv2.INTER_AREA).ravel()
            local = float(np.sort(cells)[-4:].mean())
            hj = float(0.5 * np.abs(hist - self.prev[1]).sum())
            base = float(np.median(self.recent)) if self.recent else pix
            base_l = float(np.median(self.recent_local)) if self.recent_local else local
            spike = pix >= self.cfg.min_pixel_change and pix >= self.cfg.spike_ratio * max(base, 1e-4)
            spike_l = local >= self.cfg.min_local_change and local >= self.cfg.local_spike_ratio * max(base_l, 1e-3)
            cut = (spike or spike_l or hj >= self.cfg.hist_jump) and (t - self.last_cut_t) >= self.cfg.min_shot_s
            self.scores.append((t, pix, local, hj))
            # A cut frame must not inflate the baseline for the next shot.
            if not cut:
                self.recent.append(pix)
                self.recent_local.append(local)
            else:
                self.recent.clear()
                self.recent_local.clear()
        self.prev = (grey, hist)
        if cut:
            self.shot_id += 1
            self.last_cut_t = t
        return cut


def court_view_precision(image: np.ndarray, H_img_to_court: np.ndarray) -> float:
    """Share of the projected court-line pixels that land on painted lines in ``image``.

    High for a frame filmed by the calibrated camera; low for close-ups, crowd shots, graphics or a
    camera that moved since calibration.
    """
    from vision.calibration.auto import _score, line_mask
    from vision.geometry.court import sample_court_lines

    h, w = image.shape[:2]
    scale = min(1.0, 960 / w)
    work = cv2.resize(image, (round(w * scale), round(h * scale))) if scale < 1 else image
    mask = line_mask(work)
    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    S = np.diag([scale, scale, 1.0])
    _s, precision = _score(S @ np.linalg.inv(H_img_to_court), sample_court_lines(0.10), dist, work.shape[1::-1])
    return float(max(precision, 0.0))


def cuts_from_tracks(samples, tracks, appearance=None, max_same_person=0.25, min_jump_heights=1.5) -> list[int]:
    """Sample indices where the tracker shows a cut the pixel cues missed.

    Two signatures that continuous footage essentially never produces:

    * everyone "teleports": every identity of the previous frame is gone and all current identities
      are brand new;
    * one person teleports: a track ends and, in the very next sample, a new track starts that looks
      like the same person (shirt-colour histogram) but more than ``min_jump_heights`` body heights
      away — about 2.5 m in a fraction of a second.

    Returns sample indices that should start a new shot (excluding existing shot starts).
    """
    from vision.tracking.appearance import appearance_distance

    if tracks.empty:
        return []
    tr = tracks.reset_index(drop=True)
    by_sample = {int(k): g for k, g in tr.groupby("sample_index")}
    first_seen = tr.groupby("track_id")["sample_index"].min()
    last_seen = tr.groupby("track_id")["sample_index"].max()
    shot = dict(zip(samples["sample_index"], samples["shot"], strict=True))
    order = samples.sort_values("sample_index")["sample_index"].to_numpy()
    cuts = []
    for prev_si, si in zip(order[:-1], order[1:], strict=True):
        if shot.get(si) != shot.get(prev_si):
            continue  # already a cut
        prev = by_sample.get(int(prev_si))
        cur = by_sample.get(int(si))
        if prev is None or cur is None:
            continue
        prev_ids = set(prev["track_id"])
        cur_ids = set(cur["track_id"])
        if not (prev_ids & cur_ids) and all(first_seen[i] == si for i in cur_ids):
            cuts.append(int(si))
            continue
        if appearance is None:
            continue
        ended = prev[prev["track_id"].map(last_seen) == prev_si]
        born = cur[cur["track_id"].map(first_seen) == si]
        found = False
        for ea in ended.itertuples():
            h = max(ea.y2 - ea.y1, 1.0)
            pa = np.array([(ea.x1 + ea.x2) / 2, ea.y2])
            for nb in born.itertuples():
                pb = np.array([(nb.x1 + nb.x2) / 2, nb.y2])
                if np.linalg.norm(pa - pb) <= min_jump_heights * h:
                    continue
                if appearance_distance(appearance[ea.Index], appearance[nb.Index]) <= max_same_person:
                    found = True
                    break
            if found:
                break
        if found:
            cuts.append(int(si))
    return cuts
