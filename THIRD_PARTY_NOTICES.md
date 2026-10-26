# Third-party software and models

BadmintonVision AI's own code is MIT-licensed. It depends on the following components, each under
its own licence. Model weights are **not** stored in this repository; `scripts/download_models.py`
fetches them from the publishers' official locations.

## Models

| Model | Publisher | Licence | Used for |
|---|---|---|---|
| YOLO11n / YOLO11s (`yolo11*.pt`) | Ultralytics | **AGPL-3.0** (commercial licence available from Ultralytics) | Person detection (default when the `yolo` extra is installed) |
| Pose Landmarker lite / full / heavy (`pose_landmarker_*.task`) | Google MediaPipe | Apache-2.0 | 33-point body pose |
| EfficientDet-Lite0 (`efficientdet_lite0.tflite`) | Google MediaPipe | Apache-2.0 | Alternative person detector (no AGPL dependency) |

The Pose Landmarker model card lists its training data and known limitations:
https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

## Libraries (main runtime dependencies)

| Library | Licence |
|---|---|
| OpenCV (`opencv-python-headless`) | Apache-2.0 (bundled FFmpeg: LGPL-2.1) |
| MediaPipe | Apache-2.0 |
| Ultralytics (optional `yolo` extra) | AGPL-3.0 |
| PyTorch (optional, via Ultralytics) | BSD-3-Clause |
| NumPy, SciPy, pandas, scikit-learn | BSD-3-Clause |
| Shapely | BSD-3-Clause |
| Matplotlib | Matplotlib licence (PSF-based) |
| PyArrow | Apache-2.0 |
| FastAPI, Starlette, Pydantic, Uvicorn | MIT / BSD-3-Clause |
| SQLAlchemy, Alembic | MIT |
| React, React Router, Recharts, Vite, Tailwind CSS, lucide-react | MIT / ISC |

## Algorithms re-implemented from papers

* **ByteTrack** — Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*,
  ECCV 2022. Re-implemented in `backend/vision/tracking/bytetrack.py`; no code was copied.
* **IDF1** — Ristani et al., *Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking*, 2016.
* **Log dimensionless jerk** — Balasubramanian et al., *On the analysis of movement smoothness*, J. NeuroEng. Rehabil. 2015.
* **Court dimensions** — Badminton World Federation, *Laws of Badminton*, Appendix 1.

## AGPL note

If you install the `yolo` extra and distribute the software or run it as a network service for
others, the AGPL-3.0 terms of Ultralytics apply to the combined work. For an AGPL-free deployment,
install without the extra (`pip install .`), download models with `--no-yolo`, and the pipeline uses
the MediaPipe detector automatically (`detector: auto`).
