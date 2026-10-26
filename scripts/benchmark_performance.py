"""Measure processing speed, memory and API latency on this machine.

    python scripts/benchmark_performance.py

Writes benchmarks/performance/results.{json,md}. Every number is measured on the machine the
script runs on; nothing is estimated. Real-model timings use footage built from the Ultralytics
sample photo ``zidane.jpg`` (two real people, panned to create motion) because detector/pose cost
depends on image content only weakly; speed on your footage will be similar per frame.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
MODELS = ROOT / "models" / "weights"


class PeakRSS:
    """Samples the process resident set size in a background thread."""

    def __init__(self, interval: float = 0.05):
        self.proc = psutil.Process()
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()

    def __enter__(self):
        self.base = self.proc.memory_info().rss
        self.peak = self.base
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        self.t.join()


def device_info() -> dict:
    info = {
        "cpu": platform.processor(),
        "cores": psutil.cpu_count(logical=False),
        "threads": psutil.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cuda": False,
        "gpu": None,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if info["cuda"]:
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def people_clip(path: Path, seconds: float = 20.0, fps: float = 30.0, size=(1280, 720)) -> Path:
    import ultralytics

    img = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "zidane.jpg"))
    h, w = img.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    n = int(seconds * fps)
    for k in range(n):
        x0 = int(40 * (1 + np.sin(2 * np.pi * k / n)))
        writer.write(cv2.resize(img[:, x0 : x0 + int(w * 0.93)], size))
    writer.release()
    return path


def time_component(fn, n: int, warmup: int = 3) -> dict:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    ts = np.array(ts) * 1000
    return {"mean_ms": float(ts.mean()), "p50_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)), "n": n}


def components() -> dict:
    import ultralytics
    from vision.detection.backends import MediaPipeDetector, YoloDetector
    from vision.pose.estimator import MediaPipePoseEstimator
    from vision.video.reader import Frame

    img = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "zidane.jpg"))
    frame = Frame(0, 0, 0.0, cv2.resize(img, (1280, 720)), 1.0)
    out = {}
    for weights, imgsz in (("yolo11n.pt", 640), ("yolo11s.pt", 960)):
        if (MODELS / weights).exists():
            det = YoloDetector(MODELS / weights, imgsz=imgsz)
            out[f"{weights} @ {imgsz}px ({det.device})"] = time_component(lambda d=det: d.detect(frame), 20)
    if (MODELS / "efficientdet_lite0.tflite").exists():
        mp_det = MediaPipeDetector(MODELS / "efficientdet_lite0.tflite")
        out["MediaPipe EfficientDet-Lite0 (cpu)"] = time_component(lambda: mp_det.detect(frame), 20)
    det = YoloDetector(MODELS / "yolo11n.pt")
    box = max(det.detect(frame), key=lambda d: d.score).bbox
    for name in ("pose_landmarker_lite.task", "pose_landmarker_full.task", "pose_landmarker_heavy.task"):
        if (MODELS / name).exists():
            est = MediaPipePoseEstimator(MODELS / name)
            out[f"{name.split('.')[0]} on one player crop (cpu)"] = time_component(
                lambda e=est: e.estimate(frame, box), 20
            )
    return out


def end_to_end(clip: Path, work: Path) -> dict:
    from vision.detection.backends import YoloDetector
    from vision.pipeline import PipelineInputs, PipelineSettings, run_pipeline
    from vision.video.reader import iter_frames

    first = next(iter_frames(clip))
    sel = max(YoloDetector(MODELS / "yolo11n.pt").detect(first), key=lambda d: d.score).bbox
    out = {}
    for mode in ("fast", "balanced", "accurate"):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            torch = None
        inp = PipelineInputs(
            video_path=clip,
            video_hash=f"perf-{mode}-{time.time()}",
            out_dir=work / mode,
            cache_dir=work / "cache",
            models_dir=MODELS,
            settings=PipelineSettings(performance_mode=mode, detector="yolo"),
            selection={"t": 0.0, "bbox": sel.tolist()},
        )
        with PeakRSS() as mem:
            t = time.perf_counter()
            s = run_pipeline(inp)
            wall = time.perf_counter() - t
        gpu_mb = None
        if torch is not None and torch.cuda.is_available():
            gpu_mb = torch.cuda.max_memory_allocated() / 1e6
        samples = int(round(s["quality"]["target_frames"] / max(s["quality"]["target_coverage_pct"], 1e-9) * 100))
        results_mb = sum(p.stat().st_size for p in (work / mode).glob("*")) / 1e6
        out[mode] = {
            "video_s": s["video"]["duration_s"],
            "analysis_fps": s["analysis"]["fps"],
            "frames_analysed": samples,
            "wall_s": wall,
            "realtime_factor": wall / s["video"]["duration_s"],
            "frames_per_s": samples / wall,
            "stage_s": s["timings_s"],
            "peak_rss_mb": mem.peak / 1e6,
            "rss_increase_mb": (mem.peak - mem.base) / 1e6,
            "gpu_peak_mb": gpu_mb,
            "results_mb": results_mb,
            "results_mb_per_video_min": results_mb / (s["video"]["duration_s"] / 60),
        }
        print(f"  {mode}: {wall:.1f}s for {s['video']['duration_s']:.0f}s of video")
    return out


def api_latency(work: Path) -> dict:
    from app.core.config import Settings
    from app.main import create_app
    from fastapi.testclient import TestClient

    settings = Settings(data_dir=work / "api_data", models_dir=MODELS)
    out = {}
    with TestClient(create_app(settings)) as c:
        r = c.post("/api/demo", params={"variant": 1, "duration_s": 30})
        job = r.json()["job"]["id"]
        sid = r.json()["session"]["id"]
        while c.get(f"/api/jobs/{job}").json()["status"] not in ("COMPLETED", "FAILED"):
            time.sleep(0.3)
        calls = {
            "GET /sessions": ("/api/sessions", {}),
            "GET /sessions/{id}/metrics": (f"/api/sessions/{sid}/metrics", {}),
            "GET /sessions/{id}/heatmap (json)": (f"/api/sessions/{sid}/heatmap", {}),
            "GET /sessions/{id}/heatmap (interval)": (f"/api/sessions/{sid}/heatmap", {"start": 5, "end": 15}),
            "GET /sessions/{id}/trajectory": (f"/api/sessions/{sid}/trajectory", {}),
            "GET /sessions/{id}/annotations (30 s)": (f"/api/sessions/{sid}/annotations", {"start": 0, "end": 30}),
            "GET /sessions/{id}/events": (f"/api/sessions/{sid}/events", {}),
            "GET /videos/{id}/frame": (
                f"/api/videos/{c.get(f'/api/sessions/{sid}').json()['video_id']}/frame",
                {"t": 3},
            ),
        }
        for name, (url, params) in calls.items():
            out[name] = time_component(lambda u=url, p=params: c.get(u, params=p), 15, warmup=2)
    return out


def to_md(res: dict) -> str:
    d = res["device"]
    lines = [
        "# Performance benchmark",
        "",
        f"Measured {res['date']} on **{d['cpu'] or 'unknown CPU'}** ({d['cores']} cores / {d['threads']} threads, "
        f"{d['ram_gb']} GB RAM), {d['os']}, Python {d['python']}, PyTorch {d.get('torch')}; "
        f"CUDA available: **{d['cuda']}**{' (' + d['gpu'] + ')' if d['gpu'] else ''}.",
        "",
        "## Model components (per call)",
        "",
        "| Component | mean (ms) | p95 (ms) |",
        "|---|---|---|",
    ]
    for k, v in res["components"].items():
        lines.append(f"| {k} | {v['mean_ms']:.1f} | {v['p95_ms']:.1f} |")
    lines += [
        "",
        "## End-to-end pipeline (YOLO11 + ByteTrack + MediaPipe Pose, 1280×720 input, 20 s clip)",
        "",
        "| Mode | analysis fps | frames | wall time (s) | × real time | frames/s | peak RSS (MB) | GPU peak (MB) | detect (s) | pose (s) | results (MB/min of video) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k, v in res["end_to_end"].items():
        st = v["stage_s"]
        gpu = "n/a (CPU)" if v["gpu_peak_mb"] is None else f"{v['gpu_peak_mb']:.0f}"
        lines.append(
            f"| {k} | {v['analysis_fps']:.0f} | {v['frames_analysed']} | {v['wall_s']:.1f} | {v['realtime_factor']:.2f} | "
            f"{v['frames_per_s']:.1f} | {v['peak_rss_mb']:.0f} | {gpu} | {st['detecting']:.1f} | {st['pose']:.1f} | "
            f"{v['results_mb_per_video_min']:.2f} |"
        )
    lines += [
        "",
        "## API latency (TestClient, in-process, 30 s synthetic demo session)",
        "",
        "| Endpoint | p50 (ms) | p95 (ms) |",
        "|---|---|---|",
    ]
    for k, v in res["api"].items():
        lines.append(f"| {k} | {v['p50_ms']:.1f} | {v['p95_ms']:.1f} |")
    lines += [
        "",
        "GPU acceleration is enabled automatically for YOLO when a CUDA build of PyTorch is installed; "
        "MediaPipe runs on CPU in Python on Windows/Linux. Numbers above reflect the device shown.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="bv_perf_"))
    res = {"date": time.strftime("%Y-%m-%d"), "device": device_info()}
    print("components...")
    res["components"] = components()
    print("end to end...")
    clip = people_clip(work / "people.mp4")
    res["end_to_end"] = end_to_end(clip, work)
    print("api latency...")
    res["api"] = api_latency(work)
    out = ROOT / "benchmarks" / "performance"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "results.md").write_text(to_md(res), encoding="utf-8")
    print(to_md(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
