import type { AnalyzeRequest, FrameMetrics, PlayerContext } from "../shared/types";

export const SYSTEM_PROMPT = `You are an experienced badminton coach (BWF Level 3 equivalent) reviewing a short video of a player. Your job is to give feedback the player can act on at their next session.

## What you receive
- Key frames pulled from the video, in time order. Each has an ID drawn in its top-left corner. A pose-estimation skeleton is drawn on top: the hitting arm is orange, the rest of the body is cyan.
- Measurements from that pose model for each key frame and for the whole clip (joint angles, stance width, wrist speed, movement).
- The player's own context: handedness, self-rated level, what they want to work on, and any notes.

## How to treat the measurements
The numbers come from a single camera and a general-purpose pose model. Expect roughly ±10-15° error on joint angles, worse when a limb is foreshortened, blurred, or hidden. Speeds are relative, not absolute. Use the numbers to support what you can see in the frames, not to override it. If the frames and numbers disagree, trust the frames and say so. If the skeleton is plainly misplaced in a frame, ignore that frame's numbers.

## Useful reference points (guidance, not hard rules)
- Ready position: knees flexed roughly 130-155°, feet about shoulder-width or wider (stance ratio ≥ 1.0), weight forward, racket up in front.
- Overhead strokes (clear, smash, drop): side-on preparation with the non-racket arm raised; contact high and slightly in front of the body with the hitting elbow near full extension (about 160°+ at contact); power from leg drive, hip and trunk rotation, then forearm rotation, rather than arm alone. A wide hip-shoulder separation range around the stroke is a sign of rotation.
- Net play and lunges: racket foot leads; front knee stays over the ankle rather than collapsing inward or far past the toes; racket is up early; the player pushes back to base after the shot.
- Footwork: split step timed with the opponent's hit, efficient chassé and cross steps, recovery toward the centre after each shot.
- Serve: stable base, relaxed grip, contact below the waist for a legal serve.

## Writing the feedback
- Tie every point to evidence: name frame IDs and, where helpful, a number.
- Prioritise. The top one or two improvements should be the changes that would help this player most right now, at their level and in their stated focus area.
- Give coaching cues a player can feel ("reach up and hit the shuttle at the top of your reach, in front of your head"), not textbook definitions.
- Pick drills that match the improvements and can be done with limited equipment where possible.
- Be honest and encouraging. Don't invent problems to fill the list, and don't pad strengths.
- If the clip doesn't show something (say, no strokes are visible, or only one side of the body), say so in the caveats and leave that area out instead of guessing.
- Address the player as "you".`;

const FOCUS_LABEL: Record<PlayerContext["focus"], string> = {
  general: "general technique",
  overhead: "overhead strokes (clears, smashes, drops)",
  net: "net play",
  serve: "serving",
  footwork: "footwork and movement",
  defense: "defence and blocks",
};

function fmt(value: number | null, unit = "", digits = 0): string {
  return value === null ? "n/a" : `${value.toFixed(digits)}${unit}`;
}

function fmtPct(value: number | null): string {
  return value === null ? "n/a" : `${Math.round(value * 100)}%`;
}

export function describeFrameMetrics(m: FrameMetrics): string {
  return [
    `elbow ${fmt(m.elbowAngle, "°")}`,
    `shoulder ${fmt(m.shoulderAngle, "°")}`,
    `knees L ${fmt(m.leftKneeAngle, "°")} / R ${fmt(m.rightKneeAngle, "°")}`,
    `stance ${fmt(m.stanceWidth, "x", 2)}`,
    `wrist above head ${fmt(m.wristHeight, " torso", 2)}`,
    `trunk lean ${fmt(m.trunkLean, "°")}`,
    `hip-shoulder sep ${fmt(m.hipShoulderSeparation, "°")}`,
    `wrist speed ${fmt(m.wristSpeed, " m/s", 1)}`,
  ].join(", ");
}

export function describeSession(req: AnalyzeRequest): string {
  const { context, summary } = req;
  const hand = summary.handAutoDetected
    ? `${summary.hand}-handed (auto-detected from arm movement; may be wrong)`
    : `${summary.hand}-handed`;

  const lines = [
    "## Player",
    `- Handedness: ${hand}`,
    `- Self-rated level: ${context.level}`,
    `- Wants to work on: ${FOCUS_LABEL[context.focus]}`,
    context.notes.trim() ? `- Notes from the player: """${context.notes.trim()}"""` : "- Notes from the player: none",
    "",
    "## Clip",
    `- ${summary.durationSec.toFixed(1)} s, ${summary.width}x${summary.height}, pose sampled at ${summary.sampleFps} fps (${summary.samples} samples)`,
    `- Body detected in ${fmtPct(summary.poseCoverage)} of samples`,
    "",
    "## Whole-clip measurements",
    `- Knee angle: median ${fmt(summary.posture.medianKneeAngle, "°")}, deepest ${fmt(summary.posture.minKneeAngle, "°")}`,
    `- Stance width (ankles / shoulders): median ${fmt(summary.posture.medianStanceWidth, "x", 2)}, widest ${fmt(summary.posture.maxStanceWidth, "x", 2)}`,
    `- Time in an athletic stance: ${fmtPct(summary.posture.athleticStanceRatio)}`,
    `- Median trunk lean: ${fmt(summary.posture.medianTrunkLean, "°")}`,
    `- Movement: ${fmt(summary.movement.totalTravelTorsoLengths, " torso lengths", 1)} travelled, peak ${fmt(summary.movement.peakBodySpeed, " torso/s", 1)}, moving ${fmtPct(summary.movement.activeRatio)} of the time`,
    "",
    `## Swing events detected from wrist speed: ${summary.strokes.length}`,
  ];

  for (const s of summary.strokes) {
    lines.push(
      `- Swing ${s.index} at ${s.timeSec.toFixed(2)} s (peak wrist speed ${s.peakWristSpeed.toFixed(1)} m/s, rotation range ${fmt(s.rotationRange, "°")})`,
      `  - at contact: ${describeFrameMetrics(s.contact)}`,
    );
    if (s.prep) lines.push(`  - preparation: ${describeFrameMetrics(s.prep)}`);
  }
  if (summary.strokes.length === 0) {
    lines.push("- None detected. The clip may show footwork only, or the hitting arm may be hidden.");
  }

  return lines.join("\n");
}

export function describeKeyFrame(frame: AnalyzeRequest["keyFrames"][number]): string {
  return `Frame ${frame.id} - t=${frame.timeSec.toFixed(2)} s - ${frame.label}\n${describeFrameMetrics(frame.metrics)}`;
}

export const FINAL_INSTRUCTION =
  "Review the frames and measurements above and write your coaching report for this player.";
