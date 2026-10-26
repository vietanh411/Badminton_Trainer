# Model card — stroke attribute classifier (template)

**Status: no trained model is distributed.** This card describes the model that
`scripts/train_stroke_classifier.py` produces. Each training run writes a filled-in machine-readable
card (`model_card.json`) next to the model in `data/models/strokes/<version>/<target>/`, including its
cross-validated metrics. Copy those numbers here only from an actual run on real labelled footage.

| Field | Value |
|---|---|
| Task | Classify detected stroke events into `shot_type` (smash, clear, drop, drive, net_shot, lift, other) and, separately, `stroke_side` (forehand, backhand) |
| Inputs | 23 features per event from 2-D pose landmarks around the detected contact time (racket-wrist position/displacement/speed in torso lengths, elbow and shoulder angles, knee flexion, stance width, torso orientation) and court context (distance from the net, lateral position, player speed) |
| Algorithm | Median imputation → RandomForestClassifier (300 trees, min_samples_leaf 2, balanced class weights) |
| Training data | Strokes labelled in the app by users, from sessions with completed analyses. Synthetic demo sessions are excluded by default |
| Evaluation | `GroupKFold` grouped by session (no recording appears in both train and test folds): macro-F1, per-class precision/recall, confusion matrix, expected calibration error |
| Decision rule | Highest-probability class if ≥ 0.6, otherwise `uncertain` |
| Versioning | `<target>-rf-<UTC timestamp>`, registered in the `model_versions` table; only an explicitly activated version is used by new analyses |

## Intended use

An experimental aid that suggests labels for review in the stroke timeline, for footage similar to the
training sessions (same camera placement, comparable resolution and frame rate).

## Not intended for

Officiating, performance grading, talent identification, or any decision about a person. Not for
footage from setups absent from the training data.

## Known limitations

* The model sees pose only, never the shuttle. Smash, clear and drop share most of their arm action
  (deception is part of the skill), so separating them from pose alone is fundamentally limited.
* Labels are only as good as the annotation protocol (`docs/annotation.md`); class imbalance (many
  clears, few smashes) affects per-class recall even with balanced weights.
* Performance on new players, venues and camera angles is unknown unless they were represented by
  held-out sessions.
* Monocular 2-D landmarks are noisier for the far-side player and for side-on views.
