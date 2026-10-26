"""API integration tests on a temporary data directory, using a synthetic video end to end."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from vision.synthetic.scene import court_keypoint_pixels

CAL_IDS = [
    "near_baseline__left_doubles",
    "near_baseline__right_doubles",
    "far_baseline__left_doubles",
    "far_baseline__right_doubles",
    "near_short_service__left_singles",
    "far_short_service__right_singles",
]


def make_client(tmp_path: Path, **overrides) -> TestClient:
    settings = Settings(
        data_dir=tmp_path / "data",
        models_dir=Path(__file__).resolve().parents[2] / "models" / "weights",
        cors_origins=["http://testserver"],
        **overrides,
    )
    return TestClient(create_app(settings))


@pytest.fixture
def client(tmp_path):
    with make_client(tmp_path) as c:
        yield c


def upload(client: TestClient, path: Path, name: str | None = None):
    with open(path, "rb") as fh:
        return client.post("/api/videos", files={"file": (name or path.name, fh, "video/mp4")})


def wait_job(client: TestClient, job_id: str, timeout: float = 120) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
            return j
        time.sleep(0.2)
    raise AssertionError("job did not finish in time")


def synthetic_session(client: TestClient, clip: dict, tmp_path: Path) -> str:
    """Upload the synthetic clip and attach its ground truth so the 'truth' detector can be used."""
    r = upload(client, clip["video"], "rally" + clip["video"].suffix)
    assert r.status_code == 201, r.text
    sid = r.json()["session"]["id"]
    vid = r.json()["video"]["id"]
    # Mark the stored video as synthetic (the demo endpoint does this itself).
    from app.db.models import Video
    from app.db.session import session_factory

    data = Path(client.app.dependency_overrides[next(iter(client.app.dependency_overrides))]().data_dir)
    dest = data / "uploads" / vid / "truth.json"
    shutil.copy(clip["truth_path"], dest)
    with session_factory()() as db:
        v = db.get(Video, vid)
        v.is_synthetic = True
        v.truth_path = f"uploads/{vid}/truth.json"
        db.commit()
    return sid


def test_health_and_system(client):
    assert client.get("/api/health").json()["status"] == "ok"
    sysinfo = client.get("/api/system").json()
    assert set(sysinfo["profiles"]) == {"fast", "balanced", "accurate"}
    court = client.get("/api/court").json()
    assert court["length_m"] == pytest.approx(13.4) and len(court["keypoints"]) == 32
    assert "/api/videos" in client.get("/openapi.json").json()["paths"]


def test_upload_valid_video(client, synthetic_clip):
    r = upload(client, synthetic_clip["video"], "../../evil name?.webm")
    assert r.status_code == 201, r.text
    v = r.json()["video"]
    assert (v["width"], v["height"]) == (640, 360)
    assert v["duration_s"] == pytest.approx(8.0, abs=0.1)
    assert ".." not in v["original_filename"] and "/" not in v["original_filename"]
    assert client.get(f"/api/videos/{v['id']}/thumbnail").status_code == 200
    fr = client.get(f"/api/videos/{v['id']}/frame", params={"t": 2.0, "max_width": 320})
    assert fr.status_code == 200 and fr.headers["content-type"] == "image/jpeg"
    assert int(fr.headers["x-image-width"]) == 320
    rng = client.get(f"/api/videos/{v['id']}/preview", headers={"Range": "bytes=0-99"})
    assert rng.status_code == 206 and len(rng.content) == 100


def test_upload_rejections(client, tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    assert upload(client, txt).status_code == 415
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 5000)
    r = upload(client, fake)
    assert r.status_code == 422 and "decod" in r.json()["detail"] or "open" in r.json()["detail"]
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    assert upload(client, empty).status_code == 415
    # Nothing is left behind in the temp dir.
    tmp_dir = Path(client.app.dependency_overrides[next(iter(client.app.dependency_overrides))]().tmp_dir)
    assert list(tmp_dir.glob("upload_*")) == []


def test_upload_size_and_duration_limits(tmp_path, synthetic_clip):
    with make_client(tmp_path / "a", max_upload_mb=0) as c:
        assert upload(c, synthetic_clip["video"]).status_code == 413
    with make_client(tmp_path / "b", max_duration_s=2.0) as c:
        r = upload(c, synthetic_clip["video"])
        assert r.status_code == 422 and "limit" in r.json()["detail"]


def test_api_key_enforced(tmp_path):
    with make_client(tmp_path, api_key="s3cret") as c:
        assert c.get("/api/health").status_code == 200
        assert c.get("/api/sessions").status_code == 401
        assert c.get("/api/system").status_code == 401
        assert c.get("/api/sessions", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/api/sessions", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_calibration_validation(client, synthetic_clip):
    r = upload(client, synthetic_clip["video"])
    sid = r.json()["session"]["id"]
    cam = synthetic_clip["scene"].camera
    px = court_keypoint_pixels(cam, CAL_IDS)
    pts = [{"id": i, "x": float(p[0]), "y": float(p[1])} for i, p in zip(CAL_IDS, px, strict=True)]
    body = {"points": pts, "image_width": 640, "image_height": 360, "frame_time_s": 1.0}
    dry = client.post(f"/api/sessions/{sid}/calibration", json={**body, "dry_run": True}).json()
    assert dry["saved"] is False and dry["mean_error_m"] < 0.01
    assert client.get(f"/api/sessions/{sid}/calibration").json() is None
    # Points given on a half-size preview are scaled to full resolution.
    half = [{"id": p["id"], "x": p["x"] / 2, "y": p["y"] / 2} for p in pts]
    saved = client.post(
        f"/api/sessions/{sid}/calibration", json={**body, "points": half, "image_width": 320, "image_height": 180}
    ).json()
    assert saved["saved"] is True and saved["quality"] == "good"
    np.testing.assert_allclose(np.array(saved["image_points"]), px, atol=1e-6)
    bad = [{"id": i, "x": 10.0 * k, "y": 10.0 * k} for k, i in enumerate(CAL_IDS[:4])]
    r = client.post(f"/api/sessions/{sid}/calibration", json={**body, "points": bad})
    assert r.status_code == 422
    dup = client.post(f"/api/sessions/{sid}/calibration", json={**body, "points": [pts[0]] * 4})
    assert dup.status_code == 422
    assert client.delete(f"/api/sessions/{sid}/calibration").status_code == 204
    assert client.get(f"/api/sessions/{sid}/calibration").json() is None


def test_player_selection_validation(client, synthetic_clip):
    sid = upload(client, synthetic_clip["video"]).json()["session"]["id"]
    assert (
        client.post(f"/api/sessions/{sid}/player-selection", json={"t": 1, "bbox": [10, 10, 5, 50]}).status_code == 422
    )
    ok = client.post(f"/api/sessions/{sid}/player-selection", json={"t": 1, "bbox": [10, 10, 50, 90]})
    assert ok.status_code == 200 and ok.json()["player_selection"]["bbox"] == [10, 10, 50, 90]


def test_truth_detector_refused_for_real_uploads(client, synthetic_clip):
    sid = upload(client, synthetic_clip["video"]).json()["session"]["id"]
    s = client.get(f"/api/sessions/{sid}").json()
    assert s["video"]["is_synthetic"] is False
    r = client.post(f"/api/sessions/{sid}/analyze", json={"detector": "truth"})
    assert r.status_code == 422


def test_full_analysis_flow(client, synthetic_clip, tmp_path):
    sid = synthetic_session(client, synthetic_clip, tmp_path)
    cam = synthetic_clip["scene"].camera
    px = court_keypoint_pixels(cam, CAL_IDS)
    client.post(
        f"/api/sessions/{sid}/calibration",
        json={
            "points": [{"id": i, "x": float(p[0]), "y": float(p[1])} for i, p in zip(CAL_IDS, px, strict=True)],
            "image_width": 640,
            "image_height": 360,
        },
    )
    people = client.get(f"/api/sessions/{sid}/detections", params={"t": 1.0}).json()
    assert people["detector"] == "truth" and len(people["people"]) == 2
    target = max(people["people"], key=lambda p: p["bbox"][3])  # near-side player is lower in the image
    client.post(f"/api/sessions/{sid}/player-selection", json={"t": people["t"], "bbox": target["bbox"]})

    job = client.post(f"/api/sessions/{sid}/analyze", json={"performance_mode": "fast"})
    assert job.status_code == 202, job.text
    done = wait_job(client, job.json()["id"])
    assert done["status"] == "COMPLETED", done
    assert done["progress"] == 1.0 and done["timings"]["total"] > 0

    s = client.get(f"/api/sessions/{sid}").json()
    assert s["status"] == "completed" and s["summary"]["distance_m"] > 1

    m = client.get(f"/api/sessions/{sid}/metrics").json()
    assert m["metrics"]["distance"]["total_m"] > 1
    assert m["quality"]["target_coverage_pct"] > 80
    truth = synthetic_clip["truth"]
    gt = np.array([f["players"][0]["court_xy"] for f in truth["frames"]])
    gt_dist = float(np.sum(np.linalg.norm(np.diff(gt[::3], axis=0), axis=1)))  # 10 fps like the fast profile
    assert m["metrics"]["distance"]["total_m"] == pytest.approx(gt_dist, rel=0.15)

    traj = client.get(f"/api/sessions/{sid}/trajectory").json()
    assert traj["calibrated"] and len(traj["points"]) > 50
    hm = client.get(f"/api/sessions/{sid}/heatmap").json()
    assert hm["available"] and sum(map(sum, hm["values"])) == pytest.approx(1.0, abs=1e-3)
    first = client.get(f"/api/sessions/{sid}/heatmap", params={"preset": "first_half"}).json()
    second = client.get(f"/api/sessions/{sid}/heatmap", params={"preset": "second_half"}).json()
    assert first["observed_s"] + second["observed_s"] == pytest.approx(hm["observed_s"], rel=0.05)
    front = client.get(f"/api/sessions/{sid}/heatmap", params={"zone": "front", "normalize": "seconds"}).json()
    assert front["observed_s"] <= hm["observed_s"]
    png = client.get(f"/api/sessions/{sid}/heatmap", params={"format": "png", "points": True})
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    svg = client.get(f"/api/sessions/{sid}/heatmap", params={"format": "svg"})
    assert svg.status_code == 200 and b"<svg" in svg.content[:500]

    ann = client.get(f"/api/sessions/{sid}/annotations", params={"start": 0, "end": 3}).json()
    assert len(ann["frames"]) >= 25 and any(f["pose"] for f in ann["frames"])
    assert client.get(f"/api/sessions/{sid}/annotations", params={"start": 0, "end": 500}).status_code == 422

    events = client.get(f"/api/sessions/{sid}/events").json()
    assert all("type" in e and "t" in e for e in events)
    tracks = client.get(f"/api/sessions/{sid}/tracks").json()
    assert any(t["is_target"] for t in tracks)

    # Stroke review loop.
    strokes = client.get(f"/api/sessions/{sid}/strokes").json()
    manual = client.post(
        f"/api/sessions/{sid}/strokes", json={"t": 1.23, "label_shot_type": "clear", "label_stroke_side": "forehand"}
    ).json()
    assert manual["source"] == "manual" and manual["review_status"] == "confirmed"
    if strokes:
        p = client.patch(f"/api/strokes/{strokes[0]['id']}", json={"review_status": "rejected"}).json()
        assert p["review_status"] == "rejected"
    assert client.patch("/api/strokes/nope", json={"review_status": "confirmed"}).status_code == 404

    # Exports.
    csv_ = client.get(f"/api/sessions/{sid}/export/positions.csv")
    assert csv_.status_code == 200 and csv_.text.startswith("# SYNTHETIC DEMO DATA")
    assert "court_x" in csv_.text.splitlines()[1]
    mj = client.get(f"/api/sessions/{sid}/export/metrics.json").json()
    assert mj["synthetic"] is True
    full = client.get(f"/api/sessions/{sid}/export/analysis.json").json()
    assert "events" in full and "strokes" in full
    assert client.get(f"/api/sessions/{sid}/export/strokes.csv").status_code == 200
    assert client.get(f"/api/sessions/{sid}/export/nope").status_code == 404
    pdf = client.get(f"/api/sessions/{sid}/report")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"

    # Evaluation against the synthetic ground truth.
    ev = client.get(f"/api/sessions/{sid}/evaluation").json()
    assert ev["source"] == "synthetic"
    assert ev["position"]["filtered"]["mae_m"] < 0.15
    assert ev["target"]["correct_share"] > 0.9
    assert ev["tracking"]["idf1"] > 0.75  # the small far-side opponent may fragment at 10 fps

    # Re-running with different analytics settings re-uses cached detections and pose.
    job2 = client.post(f"/api/sessions/{sid}/analyze", json={"performance_mode": "fast", "heatmap": {"sigma_m": 0.5}})
    assert wait_job(client, job2.json()["id"])["status"] == "COMPLETED"
    m2 = client.get(f"/api/sessions/{sid}/metrics").json()
    assert m2["analysis"]["cache"] == {"detections": True, "pose": True}
    assert m2["settings"]["heatmap"]["sigma_m"] == 0.5
    # Reviewed and manual strokes survive a re-run.
    after = client.get(f"/api/sessions/{sid}/strokes").json()
    assert any(e["id"] == manual["id"] for e in after)

    # Comparison with itself: all diffs zero.
    cmp_ = client.get("/api/compare", params={"a": sid, "b": sid}).json()
    assert all(r["diff"] in (0, None) for r in cmp_["metrics"])

    # Deleting removes files and rows.
    data_dir = Path(client.app.dependency_overrides[next(iter(client.app.dependency_overrides))]().data_dir)
    assert (data_dir / "sessions" / sid).exists()
    assert client.delete(f"/api/sessions/{sid}").status_code == 204
    assert client.get(f"/api/sessions/{sid}").status_code == 404
    assert not (data_dir / "sessions" / sid).exists()


def test_analysis_without_calibration_degrades_honestly(client, synthetic_clip, tmp_path):
    sid = synthetic_session(client, synthetic_clip, tmp_path)
    f0 = synthetic_clip["truth"]["frames"][30]["players"][0]
    client.post(f"/api/sessions/{sid}/player-selection", json={"t": 1.0, "bbox": f0["bbox"]})
    # Auto-calibration will be attempted; force failure by blanking the first frame is not possible here,
    # so check the API contract on a session whose auto-calibration succeeded OR is reported as missing.
    job = client.post(f"/api/sessions/{sid}/analyze", json={"performance_mode": "fast"}).json()
    done = wait_job(client, job["id"])
    assert done["status"] == "COMPLETED", done
    m = client.get(f"/api/sessions/{sid}/metrics").json()
    cal = m["quality"]["calibration"]
    assert cal["source"] in ("auto (unverified)", None)
    if cal["source"] is None:
        assert m["metrics"]["calibrated"] is False
        assert client.get(f"/api/sessions/{sid}/heatmap").json()["available"] is False
    else:
        assert any("verify" in w for w in cal["warnings"])


def test_cancel_job(client, synthetic_clip, tmp_path):
    sid = synthetic_session(client, synthetic_clip, tmp_path)
    f0 = synthetic_clip["truth"]["frames"][30]["players"][0]
    client.post(f"/api/sessions/{sid}/player-selection", json={"t": 1.0, "bbox": f0["bbox"]})
    job = client.post(f"/api/sessions/{sid}/analyze", json={"performance_mode": "accurate"}).json()
    r = client.post(f"/api/jobs/{job['id']}/cancel")
    assert r.status_code in (200, 409)
    done = wait_job(client, job["id"])
    assert done["status"] in ("CANCELLED", "COMPLETED")
    if done["status"] == "CANCELLED":
        assert client.get(f"/api/sessions/{sid}").json()["status"] == "cancelled"
    assert client.post(f"/api/jobs/{job['id']}/cancel").status_code == 409


def test_job_failure_is_reported(client, synthetic_clip, tmp_path):
    sid = synthetic_session(client, synthetic_clip, tmp_path)
    client.post(f"/api/sessions/{sid}/player-selection", json={"t": 1.0, "bbox": [0, 0, 5, 5]})  # nobody there
    job = client.post(f"/api/sessions/{sid}/analyze", json={"performance_mode": "fast"}).json()
    done = wait_job(client, job["id"])
    assert done["status"] == "FAILED"
    assert "match" in done["error"] or "selected" in done["error"]
    assert client.get(f"/api/sessions/{sid}").json()["status"] == "failed"
    assert client.get(f"/api/sessions/{sid}/metrics").status_code == 404


def test_sse_stream(client, synthetic_clip, tmp_path):
    sid = synthetic_session(client, synthetic_clip, tmp_path)
    f0 = synthetic_clip["truth"]["frames"][30]["players"][0]
    client.post(f"/api/sessions/{sid}/player-selection", json={"t": 1.0, "bbox": f0["bbox"]})
    job = client.post(f"/api/sessions/{sid}/analyze", json={"performance_mode": "fast"}).json()
    statuses = []
    with client.stream("GET", f"/api/jobs/{job['id']}/events") as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                statuses.append(json.loads(line[6:])["status"])
    assert statuses[-1] == "COMPLETED"


def test_unknown_ids(client):
    assert client.get("/api/sessions/does-not-exist").status_code == 404
    assert client.get("/api/videos/../../etc/passwd").status_code == 404
    assert client.get("/api/jobs/x").status_code == 404
    assert client.get("/api/compare", params={"a": "x", "b": "y"}).status_code == 404
