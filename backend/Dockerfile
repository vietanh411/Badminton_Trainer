# BadmintonVision AI — API + analysis worker.
# Build from the repository root:  docker build -f backend/Dockerfile -t badmintonvision-backend .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libGL/glib for OpenCV (ultralytics pulls the non-headless wheel), libgomp for PyTorch/MediaPipe,
# libEGL/libGLES because MediaPipe Tasks load them even when running on the CPU.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 libegl1 libgles2 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

# CPU PyTorch by default; pass --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 for CUDA.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
# Set WITH_YOLO=0 for an Apache-2.0-only image (MediaPipe detector, no Ultralytics/AGPL code).
ARG WITH_YOLO=1
COPY backend/pyproject.toml ./
COPY backend/app ./app
COPY backend/vision ./vision
COPY backend/migrations ./migrations
COPY backend/alembic.ini ./
RUN if [ "$WITH_YOLO" = "1" ]; then \
      pip install --index-url "$TORCH_INDEX" torch torchvision && pip install ".[yolo]"; \
    else \
      pip install .; \
    fi

# Model weights are baked into the image (about 85 MB) unless DOWNLOAD_MODELS=0; then mount
# a volume at /app/models/weights and run scripts/download_models.py once.
ARG DOWNLOAD_MODELS=1
COPY scripts /app/scripts
RUN mkdir -p /app/models/weights \
 && if [ "$DOWNLOAD_MODELS" = "1" ]; then \
      if [ "$WITH_YOLO" = "1" ]; then python /app/scripts/download_models.py; else python /app/scripts/download_models.py --no-yolo; fi; \
    fi

ENV BV_DATA_DIR=/data \
    BV_MODELS_DIR=/app/models/weights \
    YOLO_CONFIG_DIR=/tmp/ultralytics

RUN useradd --create-home --uid 1000 bv && mkdir -p /data && chown -R bv /data /app/models
USER bv

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"

# One worker process: analysis jobs run in its thread pool and share its job registry.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
