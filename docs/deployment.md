# Deployment

## Requirements

| | Minimum | Comfortable |
|---|---|---|
| CPU | 4 cores (x86-64 or Apple silicon) | 8 cores |
| RAM | 4 GB free | 8 GB |
| GPU | none (CPU-only works) | NVIDIA GPU with 4 GB+ for YOLO (`docker-compose.gpu.yml` or a CUDA PyTorch wheel) |
| Disk | 2 GB for dependencies + 85 MB models | plus ~1.2× the size of the uploaded videos |
| Python / Node | 3.11–3.13 / 20+ | 3.12 / 22 |

Measured speed on a laptop CPU is in `benchmarks/performance/results.md`. On CPU, expect analysis
to take longer than the video itself in *balanced* and *accurate* modes; *fast* mode is the quick look.

## Local development

```bash
# backend
cd backend
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on macOS/Linux
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision   # or a CUDA wheel
pip install -e ".[yolo,dev]"
python ../scripts/download_models.py
uvicorn app.main:app --reload --port 8000          # API docs: http://localhost:8000/docs

# frontend (second terminal)
cd frontend
npm install
npm run dev                                        # http://localhost:5173 (proxies /api to :8000)
```

The database (SQLite at `data/badmintonvision.db`) and its schema are created automatically on the
first start (Alembic migrations run at start-up). To use PostgreSQL, set
`BV_DATABASE_URL=postgresql+psycopg://…` and `pip install psycopg[binary]`.

Single-process production-style run: `cd frontend && npm run build`, then start uvicorn; the backend
serves `frontend/dist` at `/` when it exists.

## Docker

```bash
cp .env.example .env              # optional
docker compose up --build         # UI http://localhost:8080, API http://localhost:8000
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build   # NVIDIA GPU
```

* Build args: `WITH_YOLO=0` builds an Apache-2.0-only image (MediaPipe detector, no Ultralytics);
  `DOWNLOAD_MODELS=0` skips baking weights into the image; `TORCH_INDEX` selects CPU or CUDA wheels.
* Data persists in the `bv-data` volume. Ports bind to `127.0.0.1` only.
* The Docker files are written for this repository but were **not built in the development
  environment** of this project (Docker was not installed there); CI builds and tests the code
  without Docker. Treat the first `docker compose up` as a check.

## Cloud options

Video analysis is CPU/GPU-heavy and long-running; free static or serverless hosting cannot run it.
Reasonable set-ups:

| Option | How | Notes |
|---|---|---|
| Static frontend + local backend | Host `frontend/dist` anywhere (GitHub Pages, Netlify) with `VITE_API_BASE` pointing at a backend you run | Footage never leaves your machine if the backend is local |
| Single VM | `docker compose up` on a VM (e.g. 4 vCPU / 8 GB for CPU, or a T4/L4 GPU instance) behind HTTPS (Caddy / nginx + Let's Encrypt) | Set `BV_API_KEY`, keep `BV_MAX_QUEUED_JOBS` small, put `data/` on a persistent disk |
| Split API / worker | Run the API on a small instance and the analysis on a GPU worker | Needs a shared database (PostgreSQL) and shared storage (S3-compatible bucket mounted or synced); the current `JobManager` is in-process, so this requires replacing it with a queue (e.g. RQ/Celery) — a documented next step, not implemented |

Storage needs: uploads (original size) + per-session results + cache; see the measured artifact sizes
in `benchmarks/performance/results.md`. GPU memory use was not measured in this project's development
environment (CPU-only PyTorch); the performance script records it automatically when run on a CUDA machine.

## Security and privacy

* **Uploads**: streamed to a temp file with a hard size cap (`BV_MAX_UPLOAD_MB`), extension
  allow-list, container magic-byte check, and a full decode test with OpenCV; duration limit
  (`BV_MAX_DURATION_S`). Rejected files are deleted immediately.
* **Paths**: stored files are named by server-generated ids; the client's filename is only a sanitised
  display label. Every path served is resolved and checked to lie inside `BV_DATA_DIR`.
* **No shell**: no subprocess or shell command is built from user input; decoding and encoding use
  OpenCV's in-process FFmpeg.
* **Cost control**: bounded job queue and concurrency, one running analysis per session, duration and
  size limits. If the server is reachable by anyone else, **set `BV_API_KEY`** (every `/api` route
  except `/api/health` then requires `X-API-Key`) and put it behind HTTPS. Do not expose an
  unauthenticated instance to the internet.
* **Logging**: logs record job ids and error messages, not file paths of footage or frame content.
* **Where footage lives**: `data/uploads/<video_id>/` (original + thumbnail + optional browser proxy)
  and derived data under `data/sessions/<session_id>/`. Nothing is sent to third-party services; the
  only network access is the one-off model download.
* **Deletion**: deleting a session (UI or `DELETE /api/sessions/{id}`) removes its results,
  annotations and database rows, and the uploaded video when no other session uses it. Cache entries
  keyed by the video hash remain until `data/cache/` is cleared; delete that folder to purge all
  derived data. `BV_RETENTION_DAYS` is reserved for automatic expiry (not yet enforced).
* **Demo mode**: synthetic sessions are generated locally and flagged as synthetic everywhere; disable
  with `BV_ENABLE_DEMO=false` in deployments.
