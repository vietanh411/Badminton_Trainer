"""Optional BoT-SORT / ByteTrack via Ultralytics' built-in trackers (AGPL-3.0 dependency).

A fresh ``YOLO`` instance is created per analysis job; Ultralytics keeps tracker state on the
predictor object, so sharing an instance between jobs would leak identities across videos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vision.detection.base import Detection, auto_device
from vision.tracking.bytetrack import TrackOutput
from vision.video.reader import Frame


class UltralyticsTracker:
    def __init__(
        self,
        weights: str | Path,
        tracker: str = "botsort",
        conf: float = 0.10,
        imgsz: int = 640,
        device: str | None = None,
    ):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.tracker_cfg = f"{tracker}.yaml"
        self.conf = conf
        self.imgsz = imgsz
        self.device = device or auto_device()
        self.name = f"ultralytics-{tracker}"
        self._id_offset = 0
        self._max_id = 0

    def new_shot(self) -> None:
        """Reset Ultralytics' tracker state at a cut, keeping ids unique across shots."""
        predictor = getattr(self.model, "predictor", None)
        for trk in getattr(predictor, "trackers", None) or []:
            if hasattr(trk, "reset"):
                trk.reset()
        self._id_offset = self._max_id

    def step(self, frame: Frame) -> tuple[list[Detection], list[TrackOutput]]:
        res = self.model.track(
            frame.image,
            persist=True,
            tracker=self.tracker_cfg,
            classes=[0],
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return [], []
        xyxy = boxes.xyxy.cpu().numpy().astype(float)
        conf = boxes.conf.cpu().numpy().astype(float)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.full(len(xyxy), -1)
        dets = [Detection(b, float(c)) for b, c in zip(xyxy, conf, strict=True)]
        outs = [
            TrackOutput(int(i) + self._id_offset, b, float(c), True)
            for b, c, i in zip(xyxy, conf, ids, strict=True)
            if i >= 0
        ]
        if outs:
            self._max_id = max(self._max_id, *(o.track_id for o in outs))
        return dets, outs

    def close(self) -> None:
        self.model = None
