# Performance benchmark

Measured 2026-10-25 on **AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD** (8 cores / 16 threads, 16.5 GB RAM), Windows-11-10.0.26200-SP0, Python 3.12.0, PyTorch 2.14.1+cpu; CUDA available: **False**.

## Model components (per call)

| Component | mean (ms) | p95 (ms) |
|---|---|---|
| yolo11n.pt @ 640px (cpu) | 71.6 | 100.4 |
| yolo11s.pt @ 960px (cpu) | 261.1 | 311.8 |
| MediaPipe EfficientDet-Lite0 (cpu) | 54.6 | 67.7 |
| pose_landmarker_lite on one player crop (cpu) | 39.0 | 48.0 |
| pose_landmarker_full on one player crop (cpu) | 44.8 | 55.0 |
| pose_landmarker_heavy on one player crop (cpu) | 100.5 | 110.6 |

## End-to-end pipeline (YOLO11 + ByteTrack + MediaPipe Pose, 1280×720 input, 20 s clip)

| Mode | analysis fps | frames | wall time (s) | × real time | frames/s | peak RSS (MB) | GPU peak (MB) | detect (s) | pose (s) | results (MB/min of video) |
|---|---|---|---|---|---|---|---|---|---|---|
| fast | 10 | 200 | 24.2 | 1.21 | 8.3 | 701 | n/a (CPU) | 14.7 | 7.9 | 0.54 |
| balanced | 15 | 300 | 42.9 | 2.14 | 7.0 | 725 | n/a (CPU) | 25.4 | 15.9 | 0.77 |
| accurate | 30 | 600 | 196.4 | 9.82 | 3.1 | 898 | n/a (CPU) | 144.6 | 49.8 | 1.51 |

## API latency (TestClient, in-process, 30 s synthetic demo session)

| Endpoint | p50 (ms) | p95 (ms) |
|---|---|---|
| GET /sessions | 8.3 | 9.5 |
| GET /sessions/{id}/metrics | 11.2 | 14.2 |
| GET /sessions/{id}/heatmap (json) | 37.2 | 44.7 |
| GET /sessions/{id}/heatmap (interval) | 37.9 | 77.2 |
| GET /sessions/{id}/trajectory | 45.4 | 49.1 |
| GET /sessions/{id}/annotations (30 s) | 139.2 | 285.7 |
| GET /sessions/{id}/events | 12.6 | 13.2 |
| GET /videos/{id}/frame | 65.8 | 106.0 |

GPU acceleration is enabled automatically for YOLO when a CUDA build of PyTorch is installed; MediaPipe runs on CPU in Python on Windows/Linux. Numbers above reflect the device shown.
