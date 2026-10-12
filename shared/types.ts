// Types shared by the browser (which extracts pose data) and the server (which asks Claude).

export type Hand = "right" | "left";
export type HandPref = Hand | "auto";
export type Level = "beginner" | "intermediate" | "advanced";
export type Focus = "general" | "overhead" | "net" | "serve" | "footwork" | "defense";

export interface PlayerContext {
  hand: HandPref;
  level: Level;
  focus: Focus;
  notes: string;
}

/** Biomechanics for one sampled frame. Null when the relevant joints weren't visible. */
export interface FrameMetrics {
  /** Hitting-arm elbow angle in degrees (180 = fully straight). */
  elbowAngle: number | null;
  /** Angle between torso and hitting upper arm in degrees (0 = arm at side, 180 = straight up). */
  shoulderAngle: number | null;
  leftKneeAngle: number | null;
  rightKneeAngle: number | null;
  /** Distance between ankles divided by shoulder width. */
  stanceWidth: number | null;
  /** Hitting wrist height above the nose, in torso lengths (negative = below the head). */
  wristHeight: number | null;
  /** Torso tilt from vertical in degrees. */
  trunkLean: number | null;
  /** Angle between the shoulder line and the hip line, seen from above, in degrees. */
  hipShoulderSeparation: number | null;
  /** Hitting wrist speed relative to the hips, in m/s (estimated). */
  wristSpeed: number | null;
  /** Hip-centre movement speed, in torso lengths per second. */
  bodySpeed: number | null;
}

export interface StrokeEvent {
  index: number;
  timeSec: number;
  peakWristSpeed: number;
  contact: FrameMetrics;
  /** About 0.35 s before contact. */
  prep: FrameMetrics | null;
  /** Range of hip-shoulder separation in the second around the stroke, in degrees. */
  rotationRange: number | null;
}

export interface SessionSummary {
  durationSec: number;
  width: number;
  height: number;
  samples: number;
  sampleFps: number;
  /** Fraction of sampled frames where a body was detected. */
  poseCoverage: number;
  hand: Hand;
  handAutoDetected: boolean;
  strokes: StrokeEvent[];
  posture: {
    medianKneeAngle: number | null;
    minKneeAngle: number | null;
    medianStanceWidth: number | null;
    maxStanceWidth: number | null;
    /** Fraction of frames with bent knees (<160°) and feet at least shoulder-width apart. */
    athleticStanceRatio: number | null;
    medianTrunkLean: number | null;
  };
  movement: {
    totalTravelTorsoLengths: number | null;
    peakBodySpeed: number | null;
    /** Fraction of frames where the player is moving noticeably. */
    activeRatio: number | null;
  };
}

export interface KeyFrame {
  id: number;
  timeSec: number;
  label: string;
  /** JPEG, base64 without the data: prefix. Skeleton overlay is drawn in. */
  imageBase64: string;
  metrics: FrameMetrics;
}

export interface AnalyzeRequest {
  context: PlayerContext;
  summary: SessionSummary;
  keyFrames: KeyFrame[];
}
