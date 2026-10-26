# Methodology

This document explains how BadmintonVision turns a single fixed-camera video into court positions
and movement metrics, and exactly how each number is defined. Code references point to
`backend/vision/`.

## 1. Court model and coordinates

`geometry/court.py` encodes the BWF court (Laws of Badminton, Appendix 1) in metres:

* doubles court 13.40 × 6.10 m, singles width 5.18 m (side alleys 0.46 m);
* net at `y = 6.70`; short service lines 1.98 m from the net (`y = 4.72` and `8.68`);
  doubles long service lines 0.76 m inside each baseline (`y = 0.76` and `12.64`);
* centre line only between each short service line and its baseline.

Origin: near-left doubles corner as seen from a camera behind the near baseline; `x` across
(0 → 6.10), `y` along (0 → 13.40). Lines are modelled at their centres; the 40 mm line width is a
±2 cm ambiguity, far below the accuracy of monocular foot localisation.

**Player-relative half coordinates.** Zones and recovery use `(lateral, depth)`: depth is the
distance from the net (0–6.70 m) and lateral is measured from the player's *own* left singles
sideline while facing the net (0–5.18 m). The side (near/far) is decided per sample with a rolling
median so a lunge to the net does not flip it, and players who change ends between games are handled.

## 2. Calibration: image ↔ court homography

The floor is a plane, so a 3×3 homography `H` maps image pixels of floor points to court metres:
`[X, Y, 1]ᵀ ~ H [u, v, 1]ᵀ`. It is valid **only for points on the floor**.

**Manual calibration** (`calibration/homography.py`). The user clicks visible landmarks and assigns
each to one of 32 named court keypoints (line intersections and net-post bases).

* ≥ 4 correspondences are required. With 4–5 points we use the direct linear transform
  (`cv2.findHomography`, least squares); with ≥ 6 we use RANSAC (threshold 0.15 m in court space)
  and refit on the inliers.
* Rejected before fitting: duplicate landmarks, two landmarks on one pixel, three collinear points
  in a 4-point set (min triangle area < 0.25 m² in court space or < 100 px² in the image), or a
  nearly collinear larger set.
* Rejected after fitting: non-finite or singular `H`; Hartley-normalised condition number > 10⁷;
  image points on both sides of the horizon line (the mapping would fold the plane); RANSAC
  consensus below max(5, 60 %) of the points (several landmarks disagree, usually a wrong
  assignment); mean reprojection error > 0.20 m.
* Reported: per-point error in metres (image → court) and pixels (court → image), mean/max error,
  condition number, RANSAC outliers, and warnings — e.g. with exactly 4 points the fit is exact and
  the error cannot reveal mistakes; landmarks covering a small part of the court extrapolate poorly.
* Quality label: *good* (≥ 5 inliers, mean error ≤ 8 cm, no landmark rejected), *fair* (≤ 15 cm with at most
  a quarter of the landmarks rejected, or only 4 points), otherwise *poor*.
* Calibrations are stored with the image size they were made on; `Calibration.scaled_to()` adapts
  `H` to the down-scaled analysis frames.

**Automatic court detection** (`calibration/auto.py`) produces an *unverified suggestion* that the
UI loads into the manual editor:

1. Line mask: white top-hat (thin bright structures) ∧ low saturation ∧ high value (HSV).
2. Probabilistic Hough transform; segments split into near-horizontal (baselines/service lines)
   and side-line families; collinear segments merged by total-least-squares into lines ranked by
   supporting length.
3. Hypothesis search: each pair of image horizontals × pair of image side lines is matched to every
   ordered pair of model horizontals × model side lines, giving four intersections and a homography.
   Hypotheses are shortlisted with a cheap score, then scored by projecting all model lines and
   counting *distinct* image pixels that land on the mask (a homography that collapses the court onto
   one line cannot score well this way), minus a penalty for projected pixels off the mask.
4. Acceptance: ≥ 55 % of projected line pixels on the mask **and** < 15 % of the projected floor
   *between* lines on the mask. The second test rejects cluttered scenes where any hypothesis hits
   bright structure by chance (it rejects, for example, the Ultralytics `bus.jpg` street photo).
5. Refinement: detected lines are assigned one-to-one (Hungarian) to projected model lines; all
   matched intersections are refitted with RANSAC; the refinement is kept only if it does not
   reduce the line-mask score.

The pipeline additionally discards an unverified automatic calibration if it places every tracked
person off the court, and then reports the analysis as uncalibrated.

## 3. Detection and identity tracking

* **Detection** (`detection/`): Ultralytics YOLO11 (person class) by default, or MediaPipe
  EfficientDet-Lite0. Tiny boxes (< 4 % of frame height) are dropped. With a calibration, tracks
  whose median foot point lies > 2 m outside the court are removed (umpires, coaches, spectators,
  neighbouring courts). This filter is applied after tracking so that changing the calibration does
  not invalidate cached detections.
* **Tracking** (`tracking/bytetrack.py`): ByteTrack re-implemented from the paper. Constant-velocity
  Kalman filter on (centre, aspect, height); association of high-confidence detections to all tracks
  by IoU × score, then of low-confidence detections (0.1–0.5, typically partially occluded players)
  to the remaining tracked tracks; unconfirmed tracks need two hits; lost tracks survive 1.5 s.
  The output box is the detector's box, never the Kalman prediction, so no positions are invented.
  BoT-SORT (Ultralytics implementation, with camera-motion compensation) is selectable.
* **Target identity** (`tracking/target.py`): the user clicks a player on a frame. The seed track is
  the best-overlapping box (IoU ≥ 0.3, else nearest centre). When that track ends, the next fragment
  is chosen among tracks starting within 3 s that are physically reachable
  (distance ≤ 7 m/s × gap + 1 m, measured on the court if calibrated) and have a similar shirt colour
  (Bhattacharyya distance of upper-body hue–saturation histograms ≤ 0.55). Each link is stored as an
  `identity_link` event with its gap, distance, appearance distance and confidence. Inside a track,
  a sudden appearance change (> 0.45) or an implied speed > 10 m/s raises `possible_id_switch`.
  The user can override from any time ("from t the target is track k"), which re-runs only the cheap
  stages. Boxes of the target overlapping another person (IoU ≥ 0.15) are flagged `occluded`.
  Missing stretches ≥ 0.5 s become `tracking_gap` events and are **not** filled.

## 4. Pose estimation

MediaPipe Pose Landmarker (lite/full/heavy by performance mode) runs on a crop around the target box
expanded by 25 % and widened to an aspect ≥ 0.75 so raised arms and the racket stay in view.
Normalised crop coordinates are mapped back to full-frame pixels (`x·w_crop + x0`). Each landmark keeps
its visibility and presence. The default *image* running mode re-detects the person in every crop;
it is slower than *video* mode but robust to the crop moving and to tracking gaps.

Derived per-frame proxies (`pose/landmarks.py`): elbow angle, knee flexion (180° − hip-knee-ankle
angle), shoulder angle (hip-shoulder-elbow), arm elevation (wrist height above shoulder in torso
lengths), torso lean from vertical, torso orientation (shoulder width ÷ torso length: small = side-on),
stance width and hip-over-ankles offset in torso lengths. All are **2-D monocular proxies** — they
follow relative change within a session but are not calibrated anatomical angles, and nothing in the
app makes medical or injury judgements.

## 5. Court position of the player

Only floor points may go through `H`. Per frame (`trajectory/position.py`):

1. `pose_feet` — midpoint of both feet, each foot = mean of heel and toe landmarks (visibility ≥ 0.5),
   i.e. the centre of the base of support. Confidence 0.95 × mean visibility.
2. `pose_ankles` / `pose_one_foot` — when heels/toes are not visible (lower confidence; the ankle is
   ~9 cm above the floor, which biases the position slightly away from the camera).
3. `bbox` — bottom-centre of the detection box. Confidence 0.45 × detection score.

Feet outside the (slightly expanded) lower half of the box are treated as pose drift and ignored.
Hip or torso keypoints are never projected: being ~1 m above the floor, they would shift the
position by metres depending on camera height.

## 6. Trajectory processing

`trajectory/filtering.py`, applied to the per-sample raw court positions; raw values are kept next to
the filtered ones, and every rejected sample records why:

| Step | Rule (defaults) |
|---|---|
| Bounds | reject positions > 1.5 m outside the doubles court (`out_of_bounds`) |
| Confidence | reject position confidence < 0.25 (`low_confidence`) |
| Outliers | residual from a 0.6 s rolling median > max(0.6 m, 5 × MAD) (`outlier`) |
| Speed gate | step from the last accepted point faster than 8 m/s + 0.25 m (`speed_gate`); resets after 3 rejections or a long gap |
| Segments | consecutive valid samples whose gaps ≤ 0.4 s form a segment |
| Interpolation | linear, **only inside segments** (gaps ≤ 0.4 s), flagged `interpolated` |
| Smoothing | Savitzky–Golay, window 0.5 s, polynomial order 2, per segment |
| Derivatives | `np.gradient` on real timestamps per segment → velocity, speed, acceleration, tangential acceleration |

Frame sampling uses decoder timestamps with half-a-frame tolerance, so 30 → 15 fps sampling is
uniform even when container timestamps are rounded; irregular spacing (variable-frame-rate phone
video) is handled because every derivative and weight uses actual timestamps.

**Sample weights.** Each valid sample represents the time until the next sample in the same segment,
capped at 1.5 × the median frame interval. Gaps therefore contribute nothing, and a 60 fps and a
15 fps recording of the same movement produce the same time totals.

## 7. Movement metrics (`analytics/movement.py`)

| Metric | Definition |
|---|---|
| Observed time | Σ sample weights (time with a valid position) |
| Coverage | observed time ÷ video time analysed |
| Total distance | Σ ‖pᵢ₊₁ − pᵢ‖ over consecutive filtered samples **within one segment**; nothing is added across gaps. Interpolated share reported separately |
| Distance per minute | total distance ÷ observed minutes (comparable across clip lengths) |
| Mean speed | time-weighted mean of filtered speed over observed time |
| Mean speed while moving | same, restricted to "moving" samples |
| Moving / stationary | hysteresis: moving starts above 0.6 m/s and stops below 0.3 m/s |
| p95 and max filtered speed | of the smoothed speed; p95 is the robust figure to quote |
| Speed distribution | seconds per 0.5 m/s bin |
| Direction changes | a heading is *established* when speed > 0.8 m/s with a consistent direction (±30°) for 0.12 s; a change is counted when a newly established heading differs by ≥ 60° from the previous one **and** the player moved ≥ 0.4 m since the last counted change |
| Zones | 3 × 3 grid of the singles half: front = net → short service line (0–1.98 m), mid = to halfway to the baseline (1.98–4.34 m), rear (4.34–6.70 m); left/centre/right thirds of 5.18 m, in player-relative coordinates. Points up to 0.5 m outside the singles lines are clipped into edge zones; further out counts as "outside". Time and entries per zone, share of in-zone time |
| Zones visited vs. frequented | *visited*: ≥ 0.2 s in the zone; *frequented*: ≥ 5 % of in-zone time. Visiting a zone is not the same as defending it — neither metric says how well a zone was covered |
| Convex hull area | area of the convex hull of all positions (sensitive to outliers; the heatmap's 90 % area is more robust) |

### Recovery and footwork — movement proxies

Without shuttle tracking the app does not know when shots happen, so these are **movement proxies**:

* **Base position**: configurable, default lateral 2.59 m (centre) and 3.3 m from the net.
* **Excursion**: the player goes > 1.5 m from base; it starts where they left the 0.8 m return radius
  and ends when they come back within 0.8 m (or the segment ends).
* **Recovery time**: from the excursion's furthest point back to within 0.8 m of base. Median reported.
* **Time far from base**: seconds and share of observed time > 1.5 m away.
* **Path directness**: straight-line distance ÷ path length of the outbound leg (1 = perfectly direct);
  **extra distance** = (path − straight) ÷ straight.
* **Acceleration / deceleration**: 95th percentile of positive / negative tangential acceleration.
  Double differentiation amplifies noise; compare these only between sessions with the same setup.
* **Smoothness**: log dimensionless jerk of each excursion's speed profile (Balasubramanian et al.
  2015; closer to 0 = smoother). Sensitive to noise and sampling rate.
* **Positioning consistency**: weighted mean distance of "ready" samples (stationary and within
  1.5 m of base) from their centroid.

**Experimental composite footwork index** (on by default, can be disabled with
`movement.composite_enabled=false`; weights configurable):

```
score = 100 × Σ wₖ·sₖ / Σ wₖ  over available components
s_directness  = mean outbound directness                        (w = 0.3)
s_recovery    = clip(1 − (median recovery − 0.6 s) / 1.4 s, 0, 1) (w = 0.3)
s_smoothness  = clip((median LDLJ + 14) / 8, 0, 1)              (w = 0.2)
s_consistency = clip(1 − ready spread / 1.0 m, 0, 1)            (w = 0.2)
```

The normalisation constants are judgement calls, not fitted to expert ratings. The index is not a
validated measure of footwork efficiency; it is only meaningful for comparing one player's sessions
recorded with the same camera setup.

## 8. Heatmaps (`analytics/heatmap.py`)

1. Weighted 2-D histogram of filtered positions on a metric grid (default 0.10 m cells, grid extends
   1.5 m beyond the court), where each sample's weight is its observed seconds.
2. Gaussian smoothing with σ = 0.30 m (configurable, 0 = plain binning) — equivalent to a Gaussian
   kernel density estimate evaluated on the grid. Mass pushed off the grid by the kernel is
   renormalised so totals stay exact.
3. Normalisation options: **share** of observed time (sums to 1; use this to compare sessions of
   different length or frame rate), **seconds** per cell, or **density** in seconds per m².
4. Summary: smallest area containing 50 % / 90 % of observed time.

Interval (time range, first/second half) and zone (front/mid/rear) variants are computed on demand
from the stored positions. Without a valid calibration, metric heatmaps are disabled and the API and
UI say why.

## 9. Strokes

**Layer A — event detection (supported).** Landmarks are expressed relative to the hip midpoint in
torso lengths (rolling-median scale), so thresholds do not depend on distance to the camera. The
racket hand is the wrist with the higher 95th-percentile speed (or set explicitly). Its speed is
lightly smoothed (Savitzky–Golay, 5 samples) after interpolating gaps ≤ 0.15 s; candidate strokes are
peaks ≥ 6 torso lengths/s with prominence ≥ 3, at least 0.6 s apart. Each event has a window
(−0.4 s, +0.3 s), a confidence (prominence × arm visibility × a sampling-rate factor) and the
supporting signals (peak speed, wrist height vs. shoulder, elbow angle at the peak and its range).
Below 25 analysis fps the fastest swings are under-sampled; the API reports a warning.

**Layer B — attributes (experimental).** Modelled as separate attributes:

* `stroke_side` (forehand/backhand/uncertain): sign of the racket wrist's offset along the player's
  own left→right shoulder axis at the peak (works for players facing toward or away from the camera);
  *uncertain* when the offset is < 0.25 torso lengths or the player is side-on.
* `height_class` (overhead/side/underarm): highest wrist position just before contact vs. nose and hip.
* `shot_type`: `overhead_unspecified`, `drive`, `net_shot` / `lift` (underarm, split by distance from
  the net) or `unknown`. **Smash, clear and drop are never inferred from arm pose** — players disguise
  them deliberately and the distinguishing evidence (shuttle speed and trajectory) is not tracked.

**Supervised classifier (optional).** `strokes/classifier.py` builds 23 pose/context features per
event, trains one Random Forest per attribute with median imputation and balanced class weights, and
evaluates it with `GroupKFold` by session (events from one recording never appear in both train and
test). The report includes macro-F1, per-class metrics, the confusion matrix and expected calibration
error; it is saved as a model card next to the model. Predictions below 0.6 probability become
`uncertain`. No trained model ships with the repository; `scripts/train_stroke_classifier.py` trains one
from strokes labelled in the app.

## 10. Known limitations

* **Single fixed camera.** Camera motion invalidates the homography; the app does not stabilise video.
  BoT-SORT's camera-motion compensation helps tracking but not calibration.
* **Feet must be visible.** When they are not, positions fall back to the box bottom (less accurate)
  and the confidence drops; far-side players are small, so their positions are noisier.
* **Doubles and crowded scenes** work for tracking but identity linking is tuned for one target; two
  players in similar clothing on the same side may be confused, which is flagged rather than hidden.
* **Monocular pose** cannot recover true 3-D joint angles; depth ambiguity grows when limbs point at
  the camera.
* **Strokes**: detection is validated only on synthetic motion; real-footage precision/recall is
  unknown until annotated footage is evaluated (see `docs/evaluation.md`).
