# Annotation protocol

A small, carefully labelled benchmark is worth more than a large sloppy one. This protocol keeps
labels consistent across annotators and sessions.

## What can be annotated

| Item | Where | Stored as |
|---|---|---|
| Court calibration points | Setup → Calibration editor (each saved calibration is versioned) | `court_calibrations` table |
| Target player (identity) | Setup → Player selection; identity corrections on the Analysis page | session `player_selection`, `corrections` |
| Stroke timestamps and labels | Analysis → Strokes: confirm/reject detections, add missed strokes at the current video time, set shot type and side | `stroke_events` (`source`, `review_status`, `label_*`) |
| Player boxes, identities, ground-contact points, independent calibration points, reference court positions | Benchmark annotation file (below), edited with any tool and uploaded via `PUT /api/sessions/{id}/benchmark` | `data/sessions/<id>/benchmark/annotations.json` |

## Annotation file format (`badmintonvision-annotations/1`)

```json
{
  "format": "badmintonvision-annotations/1",
  "source": "manual",
  "annotator": "initials",
  "target_id": "P1",
  "frames": [
    {"t": 12.0, "people": [
      {"id": "P1", "bbox": [812, 402, 905, 688], "ground_contact": [858, 684]},
      {"id": "P2", "bbox": [610, 180, 652, 290], "ground_contact": null}
    ]}
  ],
  "calibration_points": [{"id": "near_short_service__center", "x": 640.5, "y": 512.0}],
  "court_positions": [{"t": 12.0, "court_xy": [2.10, 3.45]}],
  "strokes": [{"t": 14.32, "shot_type": "clear", "stroke_side": "forehand"}]
}
```

All pixel coordinates are in the **original video resolution**; times are in seconds from the start.

## Conventions

**Frames.** Annotate every 0.5 s (or every 15th frame at 30 fps) over at least 60 s of play per
session, plus every frame in a ±1 s window around occlusions and crossings — those are where trackers fail.

**Boxes.** Tight box around the visible person including raised arms, *excluding* the racket and
shuttle. If the person is more than 50 % occluded, still draw the full estimated extent but add
`"occluded": true`. Do not label people outside the court surround (spectators) unless they are on court.

**Identities.** One id per real person for the whole session (`P1`, `P2`, …). `target_id` is the
player being analysed.

**Ground contact.** The point on the floor midway between the centres of the two shoes when both
are on the ground; the planted shoe's centre when one foot is in the air; `null` while airborne (jump
smash) or when the feet are not visible. These clicks, mapped through an *independent* high-quality
calibration, give reference court positions.

**Calibration points.** For evaluating calibration, click landmarks **not** used to fit the session's
calibration (`independent_points` in the report). Click the centre of the line intersection at the
highest zoom available.

**Strokes.** The timestamp is the frame of racket–shuttle contact (if not visible, the frame where the
racket head reverses direction). Labels:
`shot_type ∈ {smash, clear, drop, drive, net_shot, lift, other}`, `stroke_side ∈ {forehand, backhand}`.
Serve = `other` unless your study treats it separately. If you cannot tell, leave the label empty
rather than guessing.

## Splits and leakage

Neighbouring frames of one rally are nearly identical, so frame-level random splits leak information
and overstate accuracy. Rules:

* Split by **session** (one recording) at minimum; by **player** when the claim is "works on new players";
  by **venue/camera** when the claim is "works on new setups".
* Never put events from the same session in both training and test folds — `train_stroke_model` uses
  `GroupKFold` grouped by session and refuses to train with fewer than two sessions.
* Report the split unit next to every number.

## Quality control

* A second annotator re-labels 10 % of frames/strokes; report inter-annotator agreement
  (IoU for boxes, distance for ground-contact points, Cohen's κ for stroke labels).
* Keep raw annotation files under version control (they contain no video).
* Record camera position, resolution, frame rate, lighting and players' consent with each session.
