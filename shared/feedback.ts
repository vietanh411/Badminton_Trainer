import { z } from "zod";

// The structured report Claude returns. The browser only imports the inferred type.

const frameIds = z
  .array(z.number().int())
  .describe("IDs of the key frames that show this, e.g. [2, 5]. Empty if not tied to a frame.");

export const SCORE_CATEGORIES = [
  "footwork",
  "ready_position",
  "stroke_technique",
  "power_generation",
  "balance_recovery",
] as const;

export const FeedbackSchema = z.object({
  detected_activity: z
    .string()
    .describe("What the player is doing in the clip, e.g. 'forehand overhead clears from the rear court'."),
  estimated_level: z.enum(["beginner", "intermediate", "advanced", "elite"]),
  summary: z.string().describe("2-4 sentence overall assessment, written directly to the player."),
  scores: z
    .array(
      z.object({
        category: z.enum(SCORE_CATEGORIES),
        score: z.number().int().describe("1 (needs a lot of work) to 10 (excellent) for this player's level."),
        rationale: z.string().describe("One or two sentences explaining the score."),
      }),
    )
    .describe("One entry per category. Omit a category only if the clip gives no evidence for it."),
  strengths: z.array(
    z.object({
      title: z.string(),
      detail: z.string(),
      frame_ids: frameIds,
    }),
  ),
  improvements: z
    .array(
      z.object({
        priority: z.enum(["high", "medium", "low"]),
        title: z.string().describe("Short name for the issue, e.g. 'Contact point too low'."),
        observation: z.string().describe("What you see in the video, referencing frames and numbers."),
        why_it_matters: z.string(),
        how_to_fix: z.string().describe("Concrete cue(s) the player can apply on court."),
        frame_ids: frameIds,
      }),
    )
    .describe("Ordered most important first."),
  drills: z.array(
    z.object({
      name: z.string(),
      targets: z.string().describe("Which improvement this drill addresses."),
      instructions: z.string(),
      dosage: z.string().describe("e.g. '3 sets of 10 reps' or '5 minutes'."),
    }),
  ),
  frame_notes: z
    .array(z.object({ frame_id: z.number().int(), note: z.string() }))
    .describe("A one-line coaching note for each key frame worth commenting on."),
  caveats: z
    .array(z.string())
    .describe("Limits of this analysis, e.g. camera angle, occlusion, clip length."),
});

export type Feedback = z.infer<typeof FeedbackSchema>;
export type ScoreCategory = (typeof SCORE_CATEGORIES)[number];

export interface AnalyzeResponse {
  feedback: Feedback;
  model: string;
}
