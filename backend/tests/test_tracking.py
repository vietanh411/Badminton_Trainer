import numpy as np
import pandas as pd
import pytest

from vision.detection.base import Detection, filter_detections
from vision.tracking.appearance import HIST_DIM, appearance_descriptor, appearance_distance
from vision.tracking.bytetrack import ByteTracker, iou_matrix
from vision.tracking.target import TargetSelectionError, resolve_target


def box(cx, cy, w=40, h=100):
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], float)


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 50 / 150, 0.0])


def test_two_players_keep_ids_while_crossing():
    tr = ByteTracker(fps=15)
    ids_a, ids_b = set(), set()
    for k in range(60):
        xa = 100 + 4 * k
        xb = 340 - 4 * k
        outs = tr.update([Detection(box(xa, 200), 0.9), Detection(box(xb, 260), 0.85)])
        for o in outs:
            cx = (o.bbox[0] + o.bbox[2]) / 2
            cy = (o.bbox[1] + o.bbox[3]) / 2
            (ids_a if cy < 230 else ids_b).add(o.track_id)
            assert abs(cx - (xa if cy < 230 else xb)) < 1e-6
    assert len(ids_a) == 1 and len(ids_b) == 1 and ids_a != ids_b


def test_brief_occlusion_keeps_identity():
    tr = ByteTracker(fps=15)
    seen = []
    for k in range(40):
        dets = [] if 15 <= k < 22 else [Detection(box(100 + 3 * k, 200), 0.9)]  # ~0.5 s missing
        seen += [o.track_id for o in tr.update(dets)]
    assert set(seen) == {1}


def test_long_absence_creates_new_identity():
    tr = ByteTracker(fps=15)
    ids = []
    for k in range(80):
        dets = [] if 10 <= k < 50 else [Detection(box(100, 200), 0.9)]  # 2.7 s missing > 1.5 s buffer
        ids += [o.track_id for o in tr.update(dets)]
    assert len(set(ids)) == 2


def test_low_score_detections_rescue_track():
    tr = ByteTracker(fps=15)
    ids = []
    for k in range(30):
        score = 0.3 if 10 <= k < 20 else 0.9  # partially occluded: low confidence
        ids += [o.track_id for o in tr.update([Detection(box(100 + 2 * k, 200), score)])]
    assert set(ids) == {1}
    assert len(ids) == 30


def test_reset_isolates_jobs():
    tr = ByteTracker(fps=15)
    for _ in range(5):
        tr.update([Detection(box(100, 100), 0.9)])
    tr.reset()
    out = tr.update([Detection(box(400, 100), 0.9)])
    assert out[0].track_id == 1 and len(tr.tracks) == 1


def test_trackers_do_not_share_state():
    a, b = ByteTracker(fps=15), ByteTracker(fps=15)
    a.update([Detection(box(100, 100), 0.9)])
    assert b.tracks == [] and b.update([]) == []


def test_filter_detections_size_and_court():
    H = np.diag([0.01, 0.01, 1.0])  # 1 px = 1 cm
    dets = [
        Detection(box(300, 600, h=100), 0.9),
        Detection(box(300, 600, h=5), 0.9),
        Detection(box(2000, 600, h=100), 0.9),
    ]
    kept = filter_detections(dets, frame_height=720, H=H)
    assert len(kept) == 1


def test_appearance_descriptor_distinguishes_colours():
    img = np.zeros((200, 200, 3), np.uint8)
    img[:, :100] = (200, 160, 30)
    img[:, 100:] = (40, 40, 200)
    a = appearance_descriptor(img, np.array([10, 10, 90, 190]))
    b = appearance_descriptor(img, np.array([110, 10, 190, 190]))
    assert a.shape == (HIST_DIM,)
    assert appearance_distance(a, a) < 1e-6
    assert appearance_distance(a, b) > 0.8


def _tracks_df(rows):
    df = pd.DataFrame(rows, columns=["sample_index", "frame_index", "t", "track_id", "x1", "y1", "x2", "y2", "score"])
    return df


def test_target_links_fragments_and_leaves_gaps():
    rows = []
    for k in range(60):
        t = k / 15
        if k < 20:
            rows.append((k, k, t, 1, *box(100 + k, 200), 0.9))  # target, first fragment
        elif k >= 26:
            rows.append((k, k, t, 3, *box(100 + k, 200), 0.9))  # target re-detected with a new id
        rows.append((k, k, t, 2, *box(500, 200), 0.9))  # opponent
    tracks = _tracks_df(rows)
    app = np.tile(np.eye(HIST_DIM, dtype=np.float32)[0], (len(tracks), 1))
    app[tracks["track_id"] == 2] = np.eye(HIST_DIM, dtype=np.float32)[5]
    samples = pd.DataFrame({"sample_index": range(60), "frame_index": range(60), "t": np.arange(60) / 15})
    res = resolve_target(tracks, app, samples, {"t": 0.5, "bbox": box(107, 200).tolist()})
    tid = res.frames["track_id"]
    assert res.seed_track == 1
    assert set(tid.dropna().astype(int)) == {1, 3}
    assert tid.iloc[20:26].isna().all()  # the 0.4 s gap is NOT filled with invented boxes
    assert any(e["type"] == "identity_link" and e["to_track"] == 3 for e in res.events)
    assert (res.frames.loc[res.frames["track_id"] == 3, "status"] == "linked").all()


def test_target_correction_overrides_identity():
    rows = []
    for k in range(30):
        rows.append((k, k, k / 15, 1, *box(100, 200), 0.9))
        rows.append((k, k, k / 15, 2, *box(500, 200), 0.9))
    tracks = _tracks_df(rows)
    app = np.full((len(tracks), HIST_DIM), 1 / HIST_DIM, np.float32)
    samples = pd.DataFrame({"sample_index": range(30), "frame_index": range(30), "t": np.arange(30) / 15})
    res = resolve_target(
        tracks, app, samples, {"t": 0.0, "bbox": box(100, 200).tolist()}, corrections=[{"t": 1.0, "track_id": 2}]
    )
    tid = res.frames["track_id"].astype("Int64")
    assert (tid[samples["t"] < 1.0] == 1).all()
    assert (tid[samples["t"] >= 1.0 - 1e-9] == 2).all()


def test_target_selection_must_match_someone():
    rows = [(0, 0, 0.0, 1, *box(100, 200), 0.9)]
    tracks = _tracks_df(rows)
    samples = pd.DataFrame({"sample_index": [0], "frame_index": [0], "t": [0.0]})
    with pytest.raises(TargetSelectionError):
        resolve_target(tracks, np.ones((1, HIST_DIM)) / HIST_DIM, samples, {"t": 0.0, "bbox": box(900, 600).tolist()})
