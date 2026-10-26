# Evaluation

This project separates three kinds of statements and labels each one:

* **Measured** — produced by a script in this repository on stated data; reproducible.
* **Synthetic** — measured, but on rendered footage with exact ground truth; it validates geometry,
  tracking association, trajectory processing and metric computation, not real-world model accuracy.
* **Untested** — functionality that exists but has no quantitative evaluation yet.

No accuracy number in this repository was estimated or copied from elsewhere.

## 1. Evaluation framework (implemented)

| Stage | Metrics | Code |
|---|---|---|
| Person detection | precision, recall, F1 at IoU 0.5 (one-to-one Hungarian matching per frame) | `evaluation/metrics.py::detection_pr` |
| Tracking | MOTA, ID switches (CLEAR-MOT), IDF1 (global identity matching), fragments per track, longest covered run | `mot_metrics` |
| Target identity | share of annotated frames where the analysed player is the true target / someone else / missing | `target_tracking_accuracy` |
| Calibration | fit reprojection error (m, px); mapped error of *independent* annotated landmarks; rejection behaviour | `homography.py`, `benchmark.py` |
| Position | MAE, median, p90, RMSE in metres (filtered and raw), per-axis MAE, missing rate, sources | `position_errors` |
| Trajectory | segments, missing rate, interpolated rate | `benchmark.py` |
| Stroke events | precision, recall, F1 with one-to-one matching inside ±0.25 s; mean timing offset | `event_detection_prf` |
| Stroke attributes | macro-F1, accuracy, confusion matrix; expected calibration error for the trained classifier | `classification_metrics`, `classifier.py` |
| Performance | per-model latency, end-to-end wall time, frames/s, real-time factor, peak RSS, GPU peak (CUDA), artifact size, API latency | `scripts/benchmark_performance.py` |

Entry points:

* `GET /api/sessions/{id}/evaluation` (and the Evaluation tab) — scores the latest analysis against the
  session's annotation file (`docs/annotation.md`), or against the simulator's ground truth for demo sessions.
* `scripts/evaluate_session.py --session <id>` — same, as a Markdown report.
* `scripts/run_synthetic_benchmark.py` — multi-scenario synthetic benchmark.
* `scripts/benchmark_performance.py` — speed, memory and latency on the current machine.

The stroke scorer uses **every machine-detected event**, including ones a reviewer later rejected;
using review decisions would leak the ground truth into the predictions.

## 2. Synthetic benchmark (measured, synthetic)

Eight 40-second rendered scenes (1280×720, 30 fps, two players), analysed by the full pipeline. Reproduce with `python scripts/run_synthetic_benchmark.py`; raw numbers in `benchmarks/synthetic/results.json`.

> **Synthetic data.** Rendered scenes with exact ground truth. Detections and pose come from ground truth plus noise (2 px box noise, 3% random misses, scripted occlusions; 1.5 px landmark noise), so these numbers measure calibration, tracking association, identity linking, trajectory processing and metric computation — **not** the accuracy of YOLO or MediaPipe on real footage.

Generated 2026-10-25 on Windows-11-10.0.26200-SP0 / Python 3.12.0 by `scripts/run_synthetic_benchmark.py`.

| Scenario | fps | Calib. grid error (m) | Target frames correct | IDF1 (all people) | Position MAE (m) | Distance error (all / observed time) | Mean speed est / true (m/s) | Max speed est / true (m/s) | Zone share MAE (pp) | Stroke event F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline-view, manual calibration | 15 | 0.038 | 98.8% | 0.99 | 0.059 | -2.4% / -2.4% | 1.40 / 1.40 | 4.57 / 4.39 | 0.1 | 0.84 |
| baseline-view, fast player | 15 | 0.042 | 98.8% | 0.99 | 0.065 | -1.9% / -1.9% | 1.87 / 1.87 | 6.05 / 5.80 | 0.2 | 0.95 |
| baseline-view, 2 occlusions of the target | 15 | 0.031 | 83.8% | 0.70 | 0.049 | -13.4% / -2.0% | 1.58 / 1.62 | 4.62 / 4.55 | 0.5 | 0.47 |
| baseline-view, automatic calibration | 15 | 0.056 | 98.8% | 0.99 | 0.065 | -1.9% / -1.9% | 1.37 / 1.37 | 4.47 / 4.30 | 0.2 | 0.89 |
| corner-view, manual calibration | 15 | 0.046 | 98.8% | 0.99 | 0.045 | -1.6% / -1.6% | 1.56 / 1.56 | 4.46 / 4.37 | 0.3 | 0.70 |
| baseline-view, sloppy clicks (4 px) | 15 | 0.054 | 98.8% | 0.99 | 0.055 | -2.3% / -2.3% | 1.49 / 1.50 | 4.44 / 4.26 | 0.2 | 0.40 |
| baseline-view, fast mode (10 fps) | 10 | 0.054 | 95.0% | 0.78 | 0.036 | -0.8% / -0.7% | 1.56 / 1.55 | 4.47 / 4.39 | 0.1 | 0.78 |
| baseline-view, accurate mode (30 fps) | 30 | 0.055 | 95.0% | 0.98 | 0.072 | -1.9% / -1.9% | 1.57 / 1.56 | 4.40 / 4.41 | 0.1 | 0.76 |

Column notes:
* *Calib. grid error*: mean distance between true and estimated court coordinates over a 7×9 grid covering the whole court.
* *Distance error*: estimated path length vs. the true path length at the 30 fps render rate — over the whole clip, and over only the time the pipeline observed (gaps are never filled, so occlusions lower the first figure).
* *Stroke event F1*: wrist-speed peak detection vs. simulated contact times, ±0.25 s. The simulated arm motion is a simple keyframed model, so this only checks that the detector and its evaluation work.

Reading the table:

* **Calibration** with 1.5 px click noise maps the whole court with 3.1–5.5 cm mean error; 4 px clicks
  and the automatic detector give 5.4 and 5.6 cm.
* **Distance** over observed time is 0.7–2.4 % *below* the true path in every scenario (smoothing cuts
  small corners). In the occlusion scenario the whole-clip figure is −13 % because 3.7 s of movement were never seen — by design, gaps are
  reported, not filled; over observed time the error is −2 %.
* **Identity**: both occlusions were bridged by identity links with no frame assigned to the wrong person;
  the missing share equals the occluded time plus the simulated 3 % random detector drop-outs.
  IDF1 is computed on the raw *tracker* ids of all people, before identity linking. It drops in the
  occlusion scene because the target gets a new tracker id after each occlusion (the linking step then
  joins them, which IDF1 does not see), and in fast mode because the small far-side opponent fragments —
  a real ByteTrack weakness on small, fast boxes at 10 fps.
* **Max speed** is off by −0.2 % to +4.3 % (noise survives smoothing at the peaks); mean speed is within
  0.04 m/s of the truth.
* **Strokes**: F1 varies from 0.40 to 0.95 between seeds. The simulated swing is a crude keyframe model, so
  this column only demonstrates that event detection and its evaluation run end to end; it says nothing
  about real swings.

## 3. Real-model checks (measured, small)

`backend/tests/test_real_models.py` runs YOLO11n + ByteTrack + MediaPipe Pose and the MediaPipe detector
path on footage made from a real photograph of two people (Ultralytics' `zidane.jpg`, panned). It checks
that the real models track both people without identity explosions, restore pose landmarks to frame
coordinates, and that the pipeline reports *uncalibrated* rather than inventing court metrics when there is
no court. `test_auto_calibration_rejects_cluttered_non_court_photo` checks that automatic court detection
refuses two real non-court photographs (a street scene and a portrait). These are functional checks, not
accuracy estimates.

## 4. Performance (measured)

Reproduce with `python scripts/benchmark_performance.py`; raw numbers in `benchmarks/performance/results.json`.

Measured 2026-10-25 on **AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD** (8 cores / 16 threads, 16.5 GB RAM), Windows-11-10.0.26200-SP0, Python 3.12.0, PyTorch 2.14.1+cpu; CUDA available: **False**.

### Model components (per call)

| Component | mean (ms) | p95 (ms) |
|---|---|---|
| yolo11n.pt @ 640px (cpu) | 71.6 | 100.4 |
| yolo11s.pt @ 960px (cpu) | 261.1 | 311.8 |
| MediaPipe EfficientDet-Lite0 (cpu) | 54.6 | 67.7 |
| pose_landmarker_lite on one player crop (cpu) | 39.0 | 48.0 |
| pose_landmarker_full on one player crop (cpu) | 44.8 | 55.0 |
| pose_landmarker_heavy on one player crop (cpu) | 100.5 | 110.6 |

### End-to-end pipeline (YOLO11 + ByteTrack + MediaPipe Pose, 1280×720 input, 20 s clip)

| Mode | analysis fps | frames | wall time (s) | × real time | frames/s | peak RSS (MB) | GPU peak (MB) | detect (s) | pose (s) | results (MB/min of video) |
|---|---|---|---|---|---|---|---|---|---|---|
| fast | 10 | 200 | 24.2 | 1.21 | 8.3 | 701 | n/a (CPU) | 14.7 | 7.9 | 0.54 |
| balanced | 15 | 300 | 42.9 | 2.14 | 7.0 | 725 | n/a (CPU) | 25.4 | 15.9 | 0.77 |
| accurate | 30 | 600 | 196.4 | 9.82 | 3.1 | 898 | n/a (CPU) | 144.6 | 49.8 | 1.51 |

### API latency (TestClient, in-process, 30 s synthetic demo session)

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

On this laptop CPU, *fast* mode runs at about 1.2× the video's length and *balanced* at about 2.1×; *accurate*
(30 fps, YOLO11s at 960 px, heavy pose) is about 10× and is meant for short clips or a GPU. Detection
dominates the run time, so a CUDA build of PyTorch should help most, but GPU timings were **not measured**
here. The playback-annotation endpoint was vectorised after the first run (975 ms → 139 ms p50 per 30 s window).

## 5. Not yet evaluated

* **Detection, tracking, pose and position accuracy on real badminton footage.** Requires annotated video;
  none is bundled (licensing and privacy of match footage).
* **Automatic court detection on real courts** (lighting, worn lines, multi-court halls, advertising boards).
* **Stroke event detection and attributes on real swings.** The synthetic arm motion only checks the
  mechanics.
* **Sensitivity to viewpoint and occlusion on real data** (the synthetic benchmark covers two camera
  positions and scripted occlusions only).

## 6. How to run a real-footage evaluation

1. Record or obtain 3+ sessions (ideally different players and venues) with consent; note camera setup.
2. Analyse each session in the app with a careful manual calibration (≥ 6 landmarks, mean error < 8 cm).
3. Fetch a pre-annotation to speed up labelling:
   `GET /api/sessions/{id}/benchmark/template` returns the analysis' own boxes, identities, ground-contact
   points and confirmed strokes **marked as machine-generated**. Correct every item per
   `docs/annotation.md` (an uncorrected template makes the evaluation circular), add independent calibration
   points, then `PUT /api/sessions/{id}/benchmark`.
4. Review strokes in the Analysis page (confirm/reject/add, label shot type and side).
5. Run `python scripts/evaluate_session.py --session <id> --out eval_<id>.md` for each session and
   aggregate by session (report mean and range across sessions, not pooled frames).
6. To train the stroke classifier: label ≥ 20 strokes across ≥ 2 sessions, run
   `python scripts/train_stroke_classifier.py`; the session-grouped cross-validation report is printed and
   stored in the model card. Activate with `--activate` only if the held-out results justify it.
