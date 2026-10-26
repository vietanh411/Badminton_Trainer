/** Plain-language metric definitions used in tooltips (mirrors docs/methodology and vision/analytics). */
export const DEF = {
  distance:
    "Sum of distances between consecutive filtered court positions inside continuous tracking segments. Long tracking gaps and rejected jumps contribute nothing.",
  distancePerMin: "Distance divided by observed (tracked) minutes, so sessions of different length compare fairly.",
  meanSpeed: "Time-weighted mean of the filtered speed over all observed time, including standing still.",
  movingSpeed: "Mean filtered speed while the player is classified as moving (hysteresis: start above 0.6 m/s, stop below 0.3 m/s).",
  p95Speed: "95th percentile of filtered speed — a robust 'fast movement' figure that ignores the single highest sample.",
  maxSpeed:
    "Highest speed after outlier rejection and Savitzky–Golay smoothing. Noisy frame-to-frame spikes are removed first; still treat a single maximum with caution.",
  movingShare: "Share of observed time spent moving (hysteresis thresholds avoid counting jitter).",
  directionChanges:
    "Changes of established movement heading by ≥ 60° after travelling ≥ 0.4 m, with the new heading held for ≥ 0.12 s at > 0.8 m/s. Measurement noise does not reach these thresholds.",
  zonesVisited:
    "Share of the 9 singles half-court zones (front/mid/rear × left/centre/right) where the player spent ≥ 0.2 s. Visiting a zone is not the same as defending it.",
  recovery:
    "Movement proxy: time from the furthest point of an excursion (> 1.5 m from the base position) back to within 0.8 m of base. Without shuttle tracking, shot timing is unknown.",
  area90: "Smallest court area that contains 90% of the observed time in the occupancy heatmap.",
  directness: "Movement proxy: straight-line distance from leaving base to the furthest point, divided by the path actually travelled (1 = perfectly direct).",
  extraDistance: "Movement proxy: extra path length relative to the straight line for outbound movements.",
  accel: "95th percentile of tangential acceleration (speeding up) from the smoothed trajectory.",
  decel: "95th percentile of tangential deceleration (slowing down) from the smoothed trajectory.",
  smoothness: "Log dimensionless jerk of each excursion's speed profile (closer to 0 = smoother). Sensitive to noise and sampling rate — a proxy only.",
  consistency: "Mean spread (m) of 'ready' positions: stationary moments within 1.5 m of base. Lower = more consistent base positioning.",
  heatmap:
    "Each position contributes the seconds of observation it represents (not a frame count), smoothed with a Gaussian kernel. Comparable across frame rates; tracking gaps contribute nothing.",
  coverage: "Share of sampled frames in which the selected player was tracked.",
  calibration: "Mean distance (m) between the clicked landmarks mapped through the homography and their true court positions.",
  idSwitch:
    "Moments where the same tracker ID suddenly changed appearance or implied an impossible jump — review these and correct the target if needed.",
  identityLink: "When the tracker lost the player and a new ID appeared nearby with matching shirt colour, the two were linked automatically.",
  strokes:
    "Candidate hitting moments from peaks in racket-wrist speed (torso lengths per second). Attributes are transparent pose heuristics — experimental, not validated.",
} as const;
