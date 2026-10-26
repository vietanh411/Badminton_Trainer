"""Download the pretrained model weights BadmintonVision uses into models/weights/.

    python scripts/download_models.py                 # MediaPipe pose (lite/full/heavy) + detector + YOLO11n
    python scripts/download_models.py --no-yolo       # Apache-2.0 models only (no Ultralytics/AGPL weights)
    python scripts/download_models.py --yolo yolo11s.pt

Sources (official):
* MediaPipe Pose Landmarker and EfficientDet-Lite0 — Google, Apache-2.0,
  https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
* YOLO11 — Ultralytics, AGPL-3.0, downloaded through the ultralytics package's own asset helper.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models" / "weights"

MEDIAPIPE = {
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_landmarker_heavy.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "efficientdet_lite0.tflite": "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite",
}


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ok {dest.name} already present")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  get {dest.name}")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:  # noqa: S310 - fixed https URLs
        shutil.copyfileobj(r, fh)
    tmp.replace(dest)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()[:16]
    print(f"    {dest.stat().st_size / 1e6:.1f} MB  sha256:{digest}...")


def fetch_yolo(name: str) -> None:
    dest = DEST / name
    if dest.exists():
        print(f"  ok {name} already present")
        return
    try:
        from ultralytics.utils.downloads import attempt_download_asset
    except ImportError:
        print("  ! ultralytics not installed; skipping YOLO (the MediaPipe detector will be used)")
        return
    print(f"  get {name} (Ultralytics, AGPL-3.0)")
    path = Path(attempt_download_asset(str(dest)))
    if path.resolve() != dest.resolve() and path.exists():
        shutil.move(str(path), dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-yolo", action="store_true", help="skip Ultralytics YOLO weights")
    ap.add_argument("--yolo", nargs="*", default=["yolo11n.pt"], help="YOLO weight files to fetch")
    ap.add_argument("--pose", nargs="*", default=list(k for k in MEDIAPIPE if k.startswith("pose")))
    args = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Downloading models to {DEST}")
    for name in [*args.pose, "efficientdet_lite0.tflite"]:
        fetch(MEDIAPIPE[name], DEST / name)
    if not args.no_yolo:
        for name in args.yolo:
            fetch_yolo(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
