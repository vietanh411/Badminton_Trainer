"""Edited footage (highlight reels): cuts between rallies filmed by the same camera."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from vision.calibration.homography import fit_homography
from vision.evaluation.benchmark import evaluate_session, truth_to_annotations
from vision.pipeline import PipelineInputs, PipelineSettings, run_pipeline
from vision.synthetic.scene import court_keypoint_pixels, default_camera, write_edited_scene
from vision.trajectory.filtering import process_trajectory
from vision.video.reader import iter_frames
from vision.video.shots import ShotDetector

CAL_IDS = [
    "near_baseline__left_doubles",
    "near_baseline__right_doubles",
    "far_baseline__left_doubles",
    "far_baseline__right_doubles",
    "near_short_service__left_singles",
    "far_short_service__right_singles",
]
SEEDS = [31, 32, 33, 34]


@pytest.fixture(scope="module")
def edited(tmp_path_factory) -> dict:
    d = tmp_path_factory.mktemp("edited")
    video, truth_path, cuts = write_edited_scene(d / "highlights", SEEDS, segment_s=4.0, graphic_after=1)
    return {
        "video": video,
        "truth_path": truth_path,
        "truth": json.loads(truth_path.read_text()),
        "cuts": cuts,
        "dir": d,
    }


def test_pixel_cut_detector_has_no_false_cuts_and_finds_graphic(edited):
    # The flat synthetic render is a hard case for the pixel cue (tiny, untextured players); the
    # tracker-based cue covers it in the pipeline. The pixel cue must still never fire falsely.
    det = ShotDetector()
    found = [
        f.timestamp_s
        for f in iter_frames(edited["video"], target_fps=15, max_width=640)
        if det.update(f.image, f.timestamp_s)
    ]
    for f in found:
        assert min(abs(f - c) for c in edited["cuts"]) <= 0.1 or 8.0 <= f <= 9.0, found
    assert any(abs(f - 8.0) <= 0.1 for f in found) and any(abs(f - 9.0) <= 0.1 for f in found)


def test_continuous_video_has_no_cuts(synthetic_clip):
    det = ShotDetector()
    assert not any(det.update(f.image, f.timestamp_s) for f in iter_frames(synthetic_clip["video"], target_fps=15))


def test_trajectory_never_spans_a_break():
    t = np.arange(0, 4, 0.1)
    x = np.where(t < 2, 1.0, 5.0)  # a 4 m jump at the cut
    y = np.full_like(t, 3.0)
    shot = (t >= 2).astype(int)
    tr = process_trajectory(t, x, y, np.full(len(t), 0.9), breaks=shot)
    assert tr["speed"].max() < 0.1  # no phantom sprint across the cut
    seg = tr["segment"].to_numpy()
    assert set(seg[shot == 0]).isdisjoint(set(seg[shot == 1]))
    # Without breaks the jump is either rejected or becomes a sprint; with breaks both sides survive.
    assert tr["raw_valid"].all()


@pytest.fixture(scope="module")
def analysed(edited):
    cam = default_camera(640, 360)
    cal = fit_homography(court_keypoint_pixels(cam, CAL_IDS), point_ids=CAL_IDS, image_size=(640, 360))
    f0 = edited["truth"]["frames"][30]["players"][0]
    out = edited["dir"] / "out"
    summary = run_pipeline(
        PipelineInputs(
            video_path=edited["video"],
            video_hash="edited-test",
            out_dir=out,
            cache_dir=edited["dir"] / "cache",
            models_dir=edited["dir"],
            settings=PipelineSettings(performance_mode="balanced", detector="truth"),
            calibration=cal.to_dict(),
            selection={"t": 1.0, "bbox": f0["bbox"]},
            synthetic_truth=edited["truth_path"],
        )
    )
    return {
        "summary": summary,
        "out": out,
        "positions": pd.read_parquet(out / "positions.parquet"),
        "tracks": pd.read_parquet(out / "tracks.parquet"),
        "events": json.loads((out / "events.json").read_text()),
    }


def _target_jump_at(truth: dict, cut: float) -> float:
    fr = [f for f in truth["frames"] if f["players"]]
    before = [f for f in fr if f["t"] < cut - 1e-6]
    after = [f for f in fr if f["t"] >= cut - 1e-6]
    if not before or not after:
        return 0.0
    pa = np.array(before[-1]["players"][0]["court_xy"])
    pb = np.array(after[0]["players"][0]["court_xy"])
    return float(np.linalg.norm(pa - pb))


def test_shots_are_detected_and_graphic_excluded(analysed, edited):
    shots = analysed["summary"]["quality"]["shots"]
    starts = [r["t_start"] for r in shots["details"]]
    # Every cut where the players visibly jump must start a new shot. (A cut where everyone happens to
    # stand where they stood before is invisible, and harmless: identity and positions stay continuous.)
    jumps = {c: _target_jump_at(edited["truth"], c) for c in edited["cuts"]}
    assert any(j > 0.5 for j in jumps.values())
    for c, j in jumps.items():
        if j > 0.5:
            assert min(abs(st - c) for st in starts) <= 0.1, (c, j, starts)
    for r in shots["details"]:  # everything inside the "REPLAY" graphic is excluded, nothing else
        inside = r["t_start"] >= 7.99 and r["t_end"] <= 9.0
        assert (r["status"] == "excluded") == inside, r


def test_tracker_ids_never_span_a_cut(analysed):
    assert (analysed["tracks"].groupby("track_id")["shot"].nunique() == 1).all()


def test_target_followed_across_every_cut(analysed, edited):
    q = analysed["summary"]["quality"]["shots"]
    used = [r for r in q["details"] if r["status"] != "excluded"]
    assert q["cross_cut_links"] == len(used) - 1  # the target was re-identified in every other usable shot
    assert q["target_uncertain_shots"] == 0
    ev = evaluate_session(analysed["out"], truth_to_annotations(edited["truth"]))
    assert ev["target"]["wrong_person_share"] == 0.0
    assert ev["target"]["correct_share"] > 0.9
    assert ev["position"]["filtered"]["mae_m"] < 0.15


def test_no_distance_or_speed_across_cuts(analysed, edited):
    pos = analysed["positions"]
    # Each trajectory segment lives inside one shot.
    seg = pos[pos["segment"] >= 0]
    assert (seg.groupby("segment")["shot"].nunique() == 1).all()
    # Distance matches the truth summed per rally segment, not the version with teleport jumps.
    fr = edited["truth"]["frames"]
    per_seg = {}
    for f in fr:
        if f["segment"] is not None:
            per_seg.setdefault(f["segment"], []).append(f["players"][0]["court_xy"])
    true_d = sum(float(np.sum(np.linalg.norm(np.diff(np.array(v), axis=0), axis=1))) for v in per_seg.values())
    est = analysed["summary"]["metrics"]["distance"]["total_m"]
    assert est == pytest.approx(true_d, rel=0.1)
    assert analysed["summary"]["metrics"]["speed"]["max_filtered_mps"] < 8
    # No positions during the excluded graphic.
    graphic = pos[(pos["t"] > 8.1) & (pos["t"] < 8.9)]
    assert graphic["court_x"].isna().all()
    assert set(graphic["status"]) == {"excluded_shot"}


def test_no_phantom_stroke_at_cuts(analysed):
    # Wrist speed is differentiated within each shot, so the landmark "jump" across a cut can never
    # become a speed peak: no stroke may peak on the first two samples of a shot.
    pos = analysed["positions"]
    shot = pos["shot"].to_numpy()
    strokes = [e for e in analysed["events"] if e["type"] == "stroke"]
    assert strokes
    for e in strokes:
        i = e["sample_index"]
        assert i >= 2 and shot[i] == shot[i - 1] == shot[i - 2], e
