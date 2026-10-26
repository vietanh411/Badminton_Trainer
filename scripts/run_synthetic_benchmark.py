"""Synthetic benchmark of the geometry / tracking-association / trajectory / metrics layers.

    python scripts/run_synthetic_benchmark.py            # full benchmark (~2-4 min on a laptop CPU)
    python scripts/run_synthetic_benchmark.py --quick    # 2 scenarios, for CI smoke checks

WHAT THIS MEASURES — and what it does not:
* Inputs are rendered synthetic scenes with exact ground truth. Person boxes and pose landmarks
  come from the *ground truth plus noise, random misses and scripted occlusions*, so this does
  NOT measure YOLO or MediaPipe accuracy (that needs annotated real footage; see docs/evaluation.md).
* It does measure everything downstream: calibration (manual clicks with pixel noise, and the
  automatic court detector), ByteTrack association, target identity linking through occlusions,
  foot-contact projection, trajectory filtering, and the movement metrics against ground truth.

Results are written to benchmarks/synthetic/results.{json,md}. All numbers are synthetic.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from vision.analytics.movement import compute_movement_metrics  # noqa: E402
from vision.calibration.auto import auto_calibrate  # noqa: E402
from vision.calibration.homography import fit_homography, project  # noqa: E402
from vision.evaluation.benchmark import evaluate_session, truth_to_annotations  # noqa: E402
from vision.pipeline import PipelineInputs, PipelineSettings, run_pipeline  # noqa: E402
from vision.synthetic.scene import court_keypoint_pixels, make_scene, write_scene  # noqa: E402
from vision.trajectory.filtering import TrajectoryConfig, process_trajectory  # noqa: E402

CAL_IDS = [
    "near_baseline__left_doubles",
    "near_baseline__right_doubles",
    "far_baseline__left_doubles",
    "far_baseline__right_doubles",
    "near_short_service__left_singles",
    "near_short_service__right_singles",
    "far_short_service__left_singles",
    "far_short_service__right_singles",
]
GRID = np.array([[x, y] for x in np.linspace(0, 6.1, 7) for y in np.linspace(0, 13.4, 9)])


@dataclass
class Scenario:
    name: str
    seed: int
    view: str = "baseline"
    intensity: float = 1.0
    calibration: str = "manual"  # manual | auto
    click_noise_px: float = 1.5
    occlusions: tuple = ()
    mode: str = "balanced"
    duration: float = 40.0


SCENARIOS = [
    Scenario("baseline-view, manual calibration", seed=101),
    Scenario("baseline-view, fast player", seed=102, intensity=1.3),
    Scenario("baseline-view, 2 occlusions of the target", seed=103, occlusions=((8.0, 9.2, 0), (20.0, 22.5, 0))),
    Scenario("baseline-view, automatic calibration", seed=104, calibration="auto"),
    Scenario("corner-view, manual calibration", seed=105, view="corner"),
    Scenario("baseline-view, sloppy clicks (4 px)", seed=106, click_noise_px=4.0),
    Scenario("baseline-view, fast mode (10 fps)", seed=107, mode="fast"),
    Scenario("baseline-view, accurate mode (30 fps)", seed=108, mode="accurate"),
]


def gt_metrics(truth: dict, t_samples: np.ndarray) -> dict:
    """Ground-truth metrics from exact positions, at the same sample times as the pipeline."""
    tt = np.array([f["t"] for f in truth["frames"]])
    gxy = np.array([f["players"][0]["court_xy"] for f in truth["frames"]])
    full_distance = float(np.sum(np.linalg.norm(np.diff(gxy, axis=0), axis=1)))
    idx = np.clip(np.searchsorted(tt, t_samples - 1e-4), 0, len(tt) - 1)
    g = gxy[idx]
    traj = process_trajectory(t_samples, g[:, 0], g[:, 1], np.ones(len(g)), TrajectoryConfig(smoothing_window_s=0.0))
    m, _, _ = compute_movement_metrics(traj)
    v = np.linalg.norm(np.gradient(gxy, tt, axis=0), axis=1)
    return {
        "distance_full_rate_m": full_distance,
        "distance_sampled_m": m["distance"]["total_m"],
        "mean_speed_mps": float(v.mean()),
        "max_speed_mps": float(v.max()),
        "zones": m["zones"]["share"],
        "depth_share": m["zones"]["depth_share"],
    }


def run(sc: Scenario, work: Path) -> dict:
    d = work / sc.name.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
    d.mkdir(parents=True, exist_ok=True)
    scene = make_scene(
        duration=sc.duration, fps=30.0, width=1280, height=720, seed=sc.seed, intensity=sc.intensity, view=sc.view
    )
    stem = d / "scene"
    if not (d / "scene.truth.json").exists():
        write_scene(scene, stem)
    truth_path = d / "scene.truth.json"
    truth = json.loads(truth_path.read_text())
    if sc.occlusions:
        truth["occlusions"] = [list(o) for o in sc.occlusions]
        truth_path.write_text(json.dumps(truth))
    video = next(p for p in d.glob("scene.*") if p.suffix in (".webm", ".mp4"))
    rng = np.random.default_rng(sc.seed)

    t0 = time.time()
    if sc.calibration == "manual":
        px = court_keypoint_pixels(scene.camera, CAL_IDS) + rng.normal(0, sc.click_noise_px, (len(CAL_IDS), 2))
        cal = fit_homography(px, point_ids=CAL_IDS, image_size=(1280, 720))
    else:
        res = auto_calibrate(scene.render(45))
        if not res.success:
            return {"scenario": sc.name, "error": f"auto calibration failed: {res.message}"}
        cal = res.calibration
    cal_time = time.time() - t0
    true_uv = project(np.linalg.inv(scene.camera.homography), GRID)
    cal_err = np.linalg.norm(project(cal.H, true_uv) - GRID, axis=1)

    f0 = truth["frames"][30]["players"][0]
    inp = PipelineInputs(
        video_path=video,
        video_hash=f"synthetic-{sc.seed}-{sc.view}-{sc.intensity}",
        out_dir=d / f"out_{sc.mode}",
        cache_dir=work / "cache",
        models_dir=ROOT / "models" / "weights",
        settings=PipelineSettings(performance_mode=sc.mode, detector="truth"),
        calibration=cal.to_dict(),
        selection={"t": 1.0, "bbox": f0["bbox"]},
        synthetic_truth=truth_path,
    )
    summary = run_pipeline(inp)
    m = summary["metrics"]
    import pandas as pd

    pos = pd.read_parquet(d / f"out_{sc.mode}" / "positions.parquet")
    gt = gt_metrics(truth, pos["t"].to_numpy())
    # True path length restricted to the intervals the pipeline actually observed (same segment on both
    # ends), so missing data (occlusions) is separated from measurement error.
    tt = np.array([f["t"] for f in truth["frames"]])
    gxy = np.array([f["players"][0]["court_xy"] for f in truth["frames"]])
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(gxy, axis=0), axis=1))])
    seg = pos["segment"].to_numpy()
    ts = pos["t"].to_numpy()
    ok = (seg[1:] == seg[:-1]) & (seg[1:] >= 0) & pos["court_x"].notna().to_numpy()[1:] & pos["court_x"].notna().to_numpy()[:-1]
    i0 = np.clip(np.searchsorted(tt, ts[:-1] - 1e-4), 0, len(tt) - 1)
    i1 = np.clip(np.searchsorted(tt, ts[1:] - 1e-4), 0, len(tt) - 1)
    gt_observed = float(np.sum((cum[i1] - cum[i0])[ok]))
    strokes = [
        {
            "t": e["t"],
            "source": "detected",
            "review_status": "candidate",
            "shot_type": e.get("shot_type"),
            "stroke_side": e.get("stroke_side"),
        }
        for e in json.loads((d / f"out_{sc.mode}" / "events.json").read_text())
        if e["type"] == "stroke"
    ]
    ev = evaluate_session(d / f"out_{sc.mode}", truth_to_annotations(truth, every_s=0.5), strokes)
    zone_err = np.mean([abs(m["zones"]["share"][z] - gt["zones"][z]) for z in gt["zones"]])
    return {
        "scenario": sc.name,
        "mode": sc.mode,
        "fps": summary["analysis"]["fps"],
        "duration_s": sc.duration,
        "calibration": {
            "method": sc.calibration,
            "grid_error_mean_m": float(cal_err.mean()),
            "grid_error_max_m": float(cal_err.max()),
            "fit_seconds": cal_time,
        },
        "tracking": ev["tracking"],
        "target": ev["target"],
        "position": ev["position"],
        "metrics_vs_truth": {
            "distance_est_m": m["distance"]["total_m"],
            "distance_true_m": gt["distance_full_rate_m"],
            "distance_true_at_sample_rate_m": gt["distance_sampled_m"],
            "distance_true_observed_m": gt_observed,
            "distance_error_observed_pct": 100 * (m["distance"]["total_m"] - gt_observed) / gt_observed,
            "distance_error_pct": 100
            * (m["distance"]["total_m"] - gt["distance_full_rate_m"])
            / gt["distance_full_rate_m"],
            "mean_speed_est": m["speed"]["mean_mps"],
            "mean_speed_true": gt["mean_speed_mps"],
            "max_speed_est": m["speed"]["max_filtered_mps"],
            "max_speed_true": gt["max_speed_mps"],
            "zone_share_mae_pct_points": 100 * zone_err,
        },
        "strokes": ev.get("strokes"),
        "quality": {
            k: summary["quality"][k]
            for k in ("target_coverage_pct", "identity_links", "possible_id_switches", "tracking_gaps")
        },
        "timings_s": summary["timings_s"],
    }


def fmt(v, d=2):
    return "—" if v is None else f"{v:.{d}f}"


def to_markdown(results: list[dict]) -> str:
    lines = [
        "# Synthetic benchmark results",
        "",
        "> **Synthetic data.** Rendered scenes with exact ground truth. Detections and pose come from ground truth "
        "plus noise (2 px box noise, 3% random misses, scripted occlusions; 1.5 px landmark noise), so these numbers "
        "measure calibration, tracking association, identity linking, trajectory processing and metric computation — "
        "**not** the accuracy of YOLO or MediaPipe on real footage.",
        "",
        f"Generated {time.strftime('%Y-%m-%d')} on {platform.platform()} / Python {platform.python_version()} by "
        "`scripts/run_synthetic_benchmark.py`.",
        "",
        "| Scenario | fps | Calib. grid error (m) | Target frames correct | IDF1 (all people) | Position MAE (m) | Distance error (all / observed time) | Mean speed est / true (m/s) | Max speed est / true (m/s) | Zone share MAE (pp) | Stroke event F1 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['scenario']} | — | {r['error']} | | | | | | | | |")
            continue
        mv = r["metrics_vs_truth"]
        lines.append(
            f"| {r['scenario']} | {r['fps']:.0f} | {fmt(r['calibration']['grid_error_mean_m'], 3)} | "
            f"{100 * r['target']['correct_share']:.1f}% | {fmt(r['tracking']['idf1'])} | "
            f"{fmt(r['position']['filtered']['mae_m'], 3)} | {mv['distance_error_pct']:+.1f}% / {mv['distance_error_observed_pct']:+.1f}% | "
            f"{fmt(mv['mean_speed_est'])} / {fmt(mv['mean_speed_true'])} | {fmt(mv['max_speed_est'])} / {fmt(mv['max_speed_true'])} | "
            f"{fmt(mv['zone_share_mae_pct_points'], 1)} | {fmt((r.get('strokes') or {}).get('events', {}).get('f1'))} |"
        )
    lines += [
        "",
        "Column notes:",
        "* *Calib. grid error*: mean distance between true and estimated court coordinates over a 7×9 grid covering the whole court.",
        "* *Distance error*: estimated path length vs. the true path length at the 30 fps render rate — over the whole clip, and "
        "over only the time the pipeline observed (gaps are never filled, so occlusions lower the first figure).",
        "* *Stroke event F1*: wrist-speed peak detection vs. simulated contact times, ±0.25 s. The simulated arm motion is a "
        "simple keyframed model, so this only checks that the detector and its evaluation work.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--work", default=str(ROOT / "data" / "benchmarks" / "synthetic"))
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    scenarios = SCENARIOS[:2] if args.quick else SCENARIOS
    if args.quick:
        for s in scenarios:
            s.duration = 15.0
    results = []
    for sc in scenarios:
        t = time.time()
        r = run(sc, work)
        print(f"{sc.name}: done in {time.time() - t:.1f}s")
        results.append(r)
    out = ROOT / "benchmarks" / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    from vision.pipeline import clean_json

    (out / "results.json").write_text(json.dumps(clean_json(results), indent=1, default=str))
    (out / "results.md").write_text(to_markdown(results), encoding="utf-8")
    print((out / "results.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
