# Synthetic benchmark results

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
