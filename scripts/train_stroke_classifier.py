"""Train the optional stroke classifier from strokes labelled in the app.

    python scripts/train_stroke_classifier.py               # train + evaluate, do not activate
    python scripts/train_stroke_classifier.py --activate    # also make it the model used by new analyses

Data: every non-rejected stroke event that has a human label (label_shot_type and/or
label_stroke_side) in sessions with a completed analysis. Synthetic demo sessions are excluded
unless --include-synthetic is given (useful only to test the pipeline mechanics).

Evaluation is session-grouped cross-validation, so events from one recording never appear in both
training and test folds. The model card with the cross-validated metrics is saved next to the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.db.models import ModelVersion, StrokeEvent, VideoSession  # noqa: E402
from app.db.session import init_engine, run_migrations, session_factory  # noqa: E402
from app.services.storage import safe_path  # noqa: E402
from sqlalchemy import select  # noqa: E402
from vision.strokes.classifier import event_features, train_stroke_model  # noqa: E402

MIN_EVENTS = 20


def collect(include_synthetic: bool) -> tuple[np.ndarray, list[dict]]:
    settings = get_settings()
    rows, feats = [], []
    with session_factory()() as db:
        sessions = db.scalars(select(VideoSession).where(VideoSession.results_dir.is_not(None))).all()
        for s in sessions:
            if s.data_source == "synthetic" and not include_synthetic:
                continue
            labelled = [
                e
                for e in db.scalars(select(StrokeEvent).where(StrokeEvent.session_id == s.id))
                if e.review_status != "rejected" and (e.label_shot_type or e.label_stroke_side)
            ]
            if not labelled:
                continue
            rd = safe_path(settings, s.results_dir)
            pos = pd.read_parquet(rd / "positions.parquet")
            pose = np.load(rd / "pose.npz")
            summary = json.loads((rd / "summary.json").read_text())
            hand = (summary["metrics"].get("strokes") or {}).get("handedness") or "right"
            t = pos["t"].to_numpy(float)
            for e in labelled:
                i = int(np.clip(np.searchsorted(t, e.t), 0, len(t) - 1))
                ctx = {
                    "depth_from_net_m": float(pos["depth_m"].iloc[i]) if "depth_m" in pos else np.nan,
                    "lateral_m": float(pos["lateral_m"].iloc[i]) if "lateral_m" in pos else np.nan,
                    "player_speed_mps": float(pos["speed"].iloc[i]) if "speed" in pos else np.nan,
                }
                feats.append(
                    event_features(t, pose["xy"].astype(float), pose["visibility"].astype(float), e.t, hand, ctx)
                )
                rows.append({"session": s.id, "shot_type": e.label_shot_type, "stroke_side": e.label_stroke_side})
    return (np.vstack(feats) if feats else np.zeros((0, 0))), rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--activate", action="store_true")
    ap.add_argument("--include-synthetic", action="store_true")
    args = ap.parse_args()
    settings = get_settings()
    settings.ensure_dirs()
    init_engine(settings)
    run_migrations(settings)
    X, rows = collect(args.include_synthetic)
    print(f"labelled events: {len(rows)} from {len({r['session'] for r in rows})} session(s)")
    if len(rows) < MIN_EVENTS:
        print(f"need at least {MIN_EVENTS} labelled strokes; label more in the Analysis page and try again")
        return 1
    version_dir = None
    trained = []
    for target in ("shot_type", "stroke_side"):
        idx = [i for i, r in enumerate(rows) if r[target]]
        labels = [rows[i][target] for i in idx]
        groups = [rows[i]["session"] for i in idx]
        if len(idx) < MIN_EVENTS or len(set(labels)) < 2 or len(set(groups)) < 2:
            print(f"skip {target}: {len(idx)} events, {len(set(labels))} classes, {len(set(groups))} sessions")
            continue
        model, report = train_stroke_model(
            X[idx],
            labels,
            groups,
            target=target,
            data_description=f"{len(idx)} labelled events from {len(set(groups))} sessions",
        )
        if version_dir is None:
            version_dir = settings.stroke_models_dir / model.version.replace(f"{target}-", "")
        model.save(version_dir / target)
        trained.append((target, model, report))
        print(f"{target}: macro-F1 (session-grouped CV) = {report['macro_f1']:.3f}  classes={report['class_counts']}")
    if not trained:
        return 1
    with session_factory()() as db:
        if args.activate:
            for mv in db.scalars(select(ModelVersion)):
                mv.is_active = False
        for target, model, report in trained:
            db.add(
                ModelVersion(
                    task=f"stroke_{target}",
                    version=model.version,
                    path=str(version_dir / target),
                    metrics={"macro_f1": report["macro_f1"]},
                    card=model.card,
                    is_active=args.activate,
                )
            )
        db.commit()
    print(f"saved to {version_dir}" + (" and activated" if args.activate else " (not active; use --activate)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
