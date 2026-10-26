"""Evaluate one analysed session against an annotation file and print a Markdown report.

    python scripts/evaluate_session.py --session <session_id> [--annotations path.json] [--out report.md]

Without --annotations the annotations saved in the app (Evaluation tab / PUT /api/sessions/{id}/benchmark)
are used; for synthetic demo sessions the simulator's ground truth is used and the report says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.db.models import StrokeEvent, Video, VideoSession  # noqa: E402
from app.db.session import init_engine, session_factory  # noqa: E402
from app.services.serialize import stroke_out  # noqa: E402
from app.services.storage import safe_path  # noqa: E402
from sqlalchemy import select  # noqa: E402
from vision.evaluation.benchmark import evaluate_session, truth_to_annotations  # noqa: E402


def fmt(v, d=3):
    if v is None:
        return "—"
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)


def to_markdown(name: str, r: dict) -> str:
    out = [f"# Evaluation — {name}", ""]
    if r["source"] == "synthetic" or r.get("synthetic"):
        out += [
            "> **Synthetic ground truth.** These numbers describe the pipeline on simulated footage, "
            "not real-world accuracy.",
            "",
        ]
    out += [f"Analysis: {r['analysis']}", ""]
    for section in ("detection", "tracking", "target", "calibration", "position", "trajectory_continuity"):
        if section not in r:
            continue
        out.append(f"## {section.replace('_', ' ').title()}")
        data = r[section]
        if section == "position":
            for k in ("filtered", "raw"):
                out.append(f"* {k}: " + ", ".join(f"{m}={fmt(v)}" for m, v in data[k].items()))
            out.append(f"* sources: {data['by_source']}")
        else:
            out += [f"* {k}: {fmt(v)}" for k, v in data.items() if not isinstance(v, (dict, list))]
        out.append("")
    if "strokes" in r:
        s = r["strokes"]
        out += [
            "## Strokes",
            "",
            "* events: " + ", ".join(f"{k}={fmt(v)}" for k, v in s["events"].items()),
            f"* stroke side macro-F1: {fmt(s['stroke_side']['macro_f1'])}",
            f"* coarse shot type macro-F1: {fmt(s['shot_type_coarse']['macro_f1'])}",
            f"* note: {s['note']}",
            "",
        ]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--annotations")
    ap.add_argument("--out")
    args = ap.parse_args()
    settings = get_settings()
    init_engine(settings)
    with session_factory()() as db:
        s = db.get(VideoSession, args.session)
        if s is None or not s.results_dir:
            print("session not found or not analysed")
            return 1
        video = db.get(Video, s.video_id)
        if args.annotations:
            ann = json.loads(Path(args.annotations).read_text())
        else:
            saved = settings.sessions_dir / s.id / "benchmark" / "annotations.json"
            if saved.exists():
                ann = json.loads(saved.read_text())
            elif video.is_synthetic and video.truth_path:
                ann = truth_to_annotations(json.loads(safe_path(settings, video.truth_path).read_text()))
            else:
                print("no annotations found for this session")
                return 1
        strokes = [stroke_out(e) for e in db.scalars(select(StrokeEvent).where(StrokeEvent.session_id == s.id))]
        report = evaluate_session(safe_path(settings, s.results_dir), ann, strokes)
        md = to_markdown(s.name, report)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
