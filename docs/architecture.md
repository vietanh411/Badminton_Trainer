# Architecture

BadmintonVision AI is a monorepo with three layers that only talk through narrow interfaces:

```mermaid
flowchart LR
    subgraph Browser["React + TypeScript SPA (frontend/)"]
        UI[Dashboard · Upload & Calibration · Analysis · Compare · Export]
    end
    subgraph API["FastAPI service (backend/app/)"]
        R[REST routes + SSE] --> S[Services: storage, results, report, compare]
        R --> J[JobManager: bounded thread pool, cancellation]
        S --> DB[(SQLite / PostgreSQL via SQLAlchemy + Alembic)]
    end
    subgraph CV["Framework-free CV core (backend/vision/)"]
        P[pipeline.py] --> V[video] & C[calibration] & D[detection] & T[tracking] & PO[pose]
        P --> TR[trajectory] & A[analytics] & ST[strokes] & VZ[visualization]
        EV[evaluation] -.scores.-> P
        SY[synthetic] -.test & demo data.-> P
    end
    UI -- "HTTP /api (JSON, JPEG, video ranges)" --> R
    J --> P
    P -- "Parquet / NPZ / JSON artifacts" --> FS[(data/ on disk)]
    S --> FS
```

* `backend/vision/` has no web or database code. Every stage is a plain function or class that can
  be unit-tested, scripted (`scripts/`) or used from a notebook.
* `backend/app/` owns persistence, HTTP, validation, security and the job lifecycle.
* `frontend/` is a static SPA. In development Vite proxies `/api` to the backend; in Docker nginx does.

## Analysis pipeline

```mermaid
flowchart TD
    U[Upload: stream to disk, size cap, magic-byte check] --> M[Probe: OpenCV metadata, decode test, thumbnail]
    M --> CAL{Calibration}
    CAL -- manual landmarks --> H[Homography fit: DLT / RANSAC + validation]
    CAL -- none --> AUTO[Auto court detection → unverified suggestion]
    AUTO --> H
    H --> DET[DETECTING: sampled frames → YOLO11 or EfficientDet → size filter]
    DET --> TRK[ByteTrack per job + appearance histograms]
    TRK --> TGT[TRACKING: court filter → seed from user's click → link fragments → corrections → switch & gap flags]
    TGT --> POSE[ESTIMATING_POSE: MediaPipe on the target crop → full-frame landmarks]
    POSE --> POS[Foot-contact point → homography → raw court positions]
    POS --> FIL[Trajectory: bounds, outliers, speed gate, short-gap interpolation, Savitzky–Golay]
    FIL --> MET[ANALYZING: distance, speed, zones, direction changes, recovery & footwork proxies]
    FIL --> HM[Time-weighted heatmap]
    POSE --> STK[Stroke candidates: wrist-speed peaks + heuristic attributes / optional classifier]
    MET & HM & STK --> OUT[RENDERING: positions.parquet, pose.npz, tracks.parquet, events.json, summary.json, heatmap.png/svg]
```

Job states exposed to the UI: `QUEUED → VALIDATING → CALIBRATING → DETECTING → TRACKING →
ESTIMATING_POSE → ANALYZING → RENDERING → COMPLETED` (or `FAILED` / `CANCELLED`). Progress is the
weighted completion of the current stage; it is written to the database at most every 0.5 s and
streamed to the browser over Server-Sent Events (`GET /api/jobs/{id}/events`), with polling as a
fallback.

### Concurrency and isolation

* Analyses run in a `ThreadPoolExecutor` (default 1 worker, `BV_MAX_CONCURRENT_JOBS`), never on the
  asyncio event loop; blocking request work (frame decoding, auto-calibration) uses `asyncio.to_thread`.
* The queue is bounded (`BV_MAX_QUEUED_JOBS`); extra requests get HTTP 429. One running job per session.
* Each job constructs its own detector, tracker and pose estimator, so no tracker state or model
  session is shared between videos. The BoT-SORT option creates a fresh Ultralytics `YOLO` object
  per job for the same reason.
* Cancellation sets a per-job `threading.Event` that frame loops check between frames.
* On start-up, jobs left in a running state by a crash are marked `FAILED` with an explanation.

### Caching

Stage outputs are cached under `data/cache/` with keys derived from the **video content hash**
(SHA-256 computed while streaming the upload) and only the settings that affect that stage:

| Cache | Key includes | Re-used when you change… |
|---|---|---|
| `det_<key>/` (samples, detections, tracks, appearance) | video hash, detector, tracker, analysis fps, width, weights, ByteTrack params | calibration, player selection, corrections, any analytics setting |
| `pose_<key>.npz` | detection key, pose model, target boxes | calibration, trajectory/metric/heatmap/stroke settings |

Everything after pose (positions, filtering, metrics, strokes, heatmaps) is recomputed on every run;
it takes well under a second for typical clips. Heatmap variants (interval, zone, smoothing,
normalisation) are computed on request from `positions.parquet`, so changing visualisation settings
never re-runs a model.

### Memory

Frames are decoded and processed one at a time (`vision/video/reader.iter_frames`), so memory does
not grow with video length. Per-frame outputs are small (boxes, 33 landmarks, one position).

## Storage layout

```
data/
  badmintonvision.db            SQLite (default; set BV_DATABASE_URL for PostgreSQL)
  uploads/<video_id>/           source.<ext>, thumbnail.jpg, proxy.webm (if transcoded), truth.json (synthetic only)
  sessions/<session_id>/
    results/<job_id>/           positions.parquet, tracks.parquet, detections.parquet, samples.parquet,
                                pose.npz, events.json, summary.json, heatmap.png, heatmap.svg
    benchmark/annotations.json  manual evaluation annotations
  cache/                        stage caches (safe to delete)
  tmp/                          in-flight uploads (deleted on success or failure)
  models/strokes/<version>/     trained stroke classifiers + model cards
```

Only the latest results directory per session is kept; older ones are deleted after a successful run.

## Data model

```mermaid
erDiagram
    VIDEOS ||--o{ VIDEO_SESSIONS : "analysed in"
    VIDEO_SESSIONS ||--o{ COURT_CALIBRATIONS : has
    VIDEO_SESSIONS ||--o{ ANALYSIS_JOBS : runs
    VIDEO_SESSIONS ||--o{ PLAYER_TRACKS : "summarises (latest run)"
    VIDEO_SESSIONS ||--o{ POSE_LANDMARKS : "points to pose.npz"
    VIDEO_SESSIONS ||--o{ COURT_POSITIONS : "points to positions.parquet"
    VIDEO_SESSIONS ||--o{ MOVEMENT_METRICS : "history per run"
    VIDEO_SESSIONS ||--o{ STROKE_EVENTS : "detected + manual, reviewed"
    VIDEO_SESSIONS ||--o{ EXPORT_ARTIFACTS : "download log"
    USERS |o--o{ VIDEO_SESSIONS : "optional owner"
    MODEL_VERSIONS
```

Bulky per-frame data lives in Parquet/NPZ files referenced from the database; videos are never
stored as BLOBs. Schema changes go through Alembic (`backend/migrations/`), applied automatically on
start-up.

## Module map

| Path | Responsibility |
|---|---|
| `vision/video/` | Container sniffing, metadata, timestamp-aware sampled decoding, thumbnails, browser proxy encoding |
| `vision/geometry/court.py` | BWF court model in metres, keypoints, lines, half-court coordinates, zones |
| `vision/calibration/` | Homography fitting/validation (`homography.py`), automatic court detection (`auto.py`) |
| `vision/detection/` | Detector protocol, court-region filtering, YOLO / MediaPipe / synthetic-truth backends |
| `vision/tracking/` | ByteTrack, Ultralytics BoT-SORT adapter, appearance descriptor, target identity resolution |
| `vision/pose/` | MediaPipe Pose on crops with coordinate restoration, angle/posture proxies |
| `vision/trajectory/` | Foot-contact positions, filtering, interpolation, smoothing, derivatives, sample weights |
| `vision/analytics/` | Movement metrics, recovery & footwork proxies, experimental composite, heatmaps |
| `vision/strokes/` | Event detection, heuristic attributes, supervised classifier + model card |
| `vision/evaluation/` | Detection/MOT/IDF1/position/event/classification metrics, annotation-file scoring |
| `vision/synthetic/` | Rendered scenes with exact ground truth (tests, demo, benchmark) |
| `vision/pipeline.py` | Stage orchestration, caching, artifact writing |
| `app/api/` | REST endpoints (OpenAPI at `/docs`) |
| `app/workers/jobs.py` | Background job manager |
| `app/services/` | Storage safety, artifact readers, PDF report, comparisons |
