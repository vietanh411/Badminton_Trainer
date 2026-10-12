import Anthropic from "@anthropic-ai/sdk";
import { z } from "zod";
import { FeedbackSchema, type Feedback } from "../shared/feedback";
import type { AnalyzeRequest } from "../shared/types";
import { describeKeyFrame, describeSession, FINAL_INSTRUCTION, SYSTEM_PROMPT } from "./prompt";

export const MODEL = process.env.CLAUDE_MODEL ?? "claude-opus-5-5";

type Effort = "low" | "medium" | "high" | "xhigh" | "max";
const EFFORTS: Effort[] = ["low", "medium", "high", "xhigh", "max"];
const EFFORT: Effort = EFFORTS.find((e) => e === process.env.CLAUDE_EFFORT) ?? "high";

/** An analysis that reached Claude but didn't produce a usable report. */
export class CoachError extends Error {}

/**
 * JSON schema for structured output. Built by hand from the zod schema because the SDK helper
 * turns `enum` into a description (unenforced); the API supports enums directly but not
 * numeric bounds, so those are stripped instead.
 */
function toStrictJsonSchema(schema: z.ZodType): { [key: string]: unknown } {
  const DROP = new Set(["$schema", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]);
  const clean = (node: unknown): unknown => {
    if (Array.isArray(node)) return node.map(clean);
    if (!node || typeof node !== "object") return node;
    const out: { [key: string]: unknown } = {};
    for (const [key, value] of Object.entries(node)) {
      if (!DROP.has(key)) out[key] = key === "properties" ? mapValues(value, clean) : clean(value);
    }
    if (out.type === "object") out.additionalProperties = false;
    return out;
  };
  const mapValues = (obj: unknown, fn: (v: unknown) => unknown) =>
    Object.fromEntries(Object.entries(obj as object).map(([k, v]) => [k, fn(v)]));
  return clean(z.toJSONSchema(schema, { reused: "inline" })) as { [key: string]: unknown };
}

const FEEDBACK_JSON_SCHEMA = toStrictJsonSchema(FeedbackSchema);

let client: Anthropic | null = null;
function getClient(): Anthropic {
  // Created lazily so the server can start (and serve the UI) before credentials are set up.
  client ??= new Anthropic();
  return client;
}

export async function analyzeSession(req: AnalyzeRequest): Promise<{ feedback: Feedback; model: string }> {
  const content: Anthropic.Beta.BetaContentBlockParam[] = [{ type: "text", text: describeSession(req) }];

  content.push({ type: "text", text: `## Key frames (${req.keyFrames.length})` });
  for (const frame of req.keyFrames) {
    content.push(
      { type: "text", text: describeKeyFrame(frame) },
      { type: "image", source: { type: "base64", media_type: "image/jpeg", data: frame.imageBase64 } },
    );
  }
  content.push({ type: "text", text: FINAL_INSTRUCTION });

  // Streamed because a request with ~10 images and high effort can run past HTTP timeouts.
  const stream = getClient().beta.messages.stream({
    model: MODEL,
    max_tokens: 32000,
    system: SYSTEM_PROMPT,
    thinking: { type: "adaptive" },
    output_config: { effort: EFFORT, format: { type: "json_schema", schema: FEEDBACK_JSON_SCHEMA } },
    // If a safety classifier declines the request, retry server-side on Anthropic's recommended model.
    betas: ["server-side-fallback-2026-07-01"],
    fallbacks: "default",
    messages: [{ role: "user", content }],
  });

  const message = await stream.finalMessage();

  if (message.stop_reason === "refusal") {
    throw new CoachError("The model declined to analyse this clip. Try a different video.");
  }
  if (message.stop_reason === "max_tokens") {
    throw new CoachError("The report was cut off before it finished. Try a shorter clip.");
  }

  // If a fallback model took over mid-response, only the text after the last switch point counts.
  const lastFallback = message.content.findLastIndex((block) => block.type === "fallback");
  const text = message.content
    .slice(lastFallback + 1)
    .flatMap((block) => (block.type === "text" ? [block.text] : []))
    .join("");

  let parsed: ReturnType<typeof FeedbackSchema.safeParse>;
  try {
    parsed = FeedbackSchema.safeParse(JSON.parse(text));
  } catch {
    throw new CoachError("The model's report couldn't be read. Please try again.");
  }
  if (!parsed.success) {
    console.error("[coach] report failed validation:", parsed.error.issues);
    throw new CoachError("The model's report was incomplete. Please try again.");
  }

  const u = message.usage;
  console.log(
    `[coach] ${message.model}: ${u.input_tokens} in (+${u.cache_read_input_tokens ?? 0} cached), ${u.output_tokens} out`,
  );

  return { feedback: parsed.data, model: message.model };
}
