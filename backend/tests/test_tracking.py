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


def test_split_tracks_at_identity_swap_but_not_at_far_side_jitter():
    from vision.calibration.homography import fit_homography
    from vision.synthetic.scene import court_keypoint_pixels, default_camera
    from vision.tracking.target import split_tracks_at_jumps

    ids = [
        "near_baseline__left_doubles",
        "near_baseline__right_doubles",
        "far_baseline__left_doubles",
        "far_baseline__right_doubles",
        "near_short_service__left_singles",
        "far_short_service__right_singles",
    ]
    cam = default_camera(1280, 720)
    H = fit_homography(court_keypoint_pixels(cam, ids), point_ids=ids).H
    rows = []
    for k in range(20):  # near player whose id is handed to the far player at k = 10
        if k < 10:
            rows.append((k, k, k / 15, 1, *box(560, 560, 70, 200), 0.9))
        else:
            rows.append((k, k, k / 15, 1, *box(700, 240, 25, 70), 0.9))
    rng = np.random.default_rng(0)
    for k in range(20):  # far player with 3 px box jitter: large in depth, small in the image
        cy = 240 + rng.normal(0, 3)
        rows.append((k, k, k / 15, 2, *box(620, cy, 25, 70), 0.9))
    tracks = _tracks_df(rows).assign(shot=0)
    out, n = split_tracks_at_jumps(tracks, {0: H})
    assert n == 1
    assert out.loc[tracks["track_id"] == 2, "track_id"].nunique() == 1
    t1 = out.loc[tracks["track_id"] == 1]
    assert t1["track_id"].iloc[:10].nunique() == 1 and t1["track_id"].iloc[10:].nunique() == 1
    assert t1["track_id"].iloc[0] != t1["track_id"].iloc[10]


def test_cross_shot_reid_is_not_confused_by_fragments_of_the_same_player():
    rows = []
    # shot 0: target (id 1) + opponent (id 2); shot 1: target split into two fragments (3, 4) + opponent (5)
    for k in range(30):
        rows.append((k, k, k / 15, 1, *box(300, 400), 0.9, 0))
        rows.append((k, k, k / 15, 2, *box(600, 150, 20, 60), 0.9, 0))
    for k in range(30, 60):
        tid = 3 if k < 45 else 4
        rows.append((k, k, k / 15, tid, *box(500 if k < 45 else 200, 400), 0.9, 1))
        rows.append((k, k, k / 15, 5, *box(650, 150, 20, 60), 0.9, 1))
    tracks = pd.DataFrame(
        rows, columns=["sample_index", "frame_index", "t", "track_id", "x1", "y1", "x2", "y2", "score", "shot"]
    )
    white, red = np.eye(HIST_DIM, dtype=np.float32)[0], np.eye(HIST_DIM, dtype=np.float32)[7]
    app = np.where(tracks["track_id"].isin([1, 3, 4]).to_numpy()[:, None], white, red)
    samples = pd.DataFrame(
        {"sample_index": range(60), "frame_index": range(60), "t": np.arange(60) / 15, "shot": [0] * 30 + [1] * 30}
    )
    res = resolve_target(tracks, app, samples, {"t": 0.5, "bbox": box(300, 400).tolist()})
    assert not any(e["type"] == "shot_target_uncertain" for e in res.events)
    later = res.frames[res.frames["shot"] == 1]["track_id"].dropna().astype(int)
    assert set(later) <= {3, 4} and 3 in set(later)
