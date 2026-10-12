import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import Anthropic from "@anthropic-ai/sdk";
import express from "express";
import type { AnalyzeRequest } from "../shared/types";
import { analyzeSession, CoachError, MODEL } from "./coach";

try {
  process.loadEnvFile();
} catch {
  // No .env file - credentials may come from the environment or `ant auth login`.
}

const PORT = Number(process.env.PORT ?? 3001);
const MAX_FRAMES = 12;

const app = express();
app.use(express.json({ limit: "40mb" }));

app.get("/api/health", (_req, res) => {
  res.json({ ok: true, model: MODEL });
});

function validate(body: unknown): string | null {
  const req = body as Partial<AnalyzeRequest> | undefined;
  if (!req || typeof req !== "object") return "Request body must be JSON.";
  if (!req.context || !req.summary) return "Missing context or summary.";
  if (!Array.isArray(req.keyFrames) || req.keyFrames.length === 0) return "No key frames were sent.";
  if (req.keyFrames.length > MAX_FRAMES) return `Send at most ${MAX_FRAMES} key frames.`;
  for (const f of req.keyFrames) {
    if (typeof f?.imageBase64 !== "string" || f.imageBase64.length < 100) return "A key frame image is missing.";
    if (f.imageBase64.startsWith("data:")) return "Key frame images must be raw base64, not data URLs.";
  }
  return null;
}

app.post("/api/analyze", async (req, res) => {
  const problem = validate(req.body);
  if (problem) {
    res.status(400).json({ error: problem });
    return;
  }

  const started = Date.now();
  try {
    const result = await analyzeSession(req.body as AnalyzeRequest);
    console.log(`[coach] analysis took ${((Date.now() - started) / 1000).toFixed(1)} s`);
    res.json(result);
  } catch (err) {
    console.error("[coach] analysis failed:", err);
    if (err instanceof CoachError) {
      res.status(422).json({ error: err.message });
    } else if (err instanceof Anthropic.AuthenticationError) {
      res.status(401).json({ error: "The Anthropic API key is missing or invalid. Set ANTHROPIC_API_KEY in .env." });
    } else if (err instanceof Anthropic.RateLimitError) {
      res.status(429).json({ error: "Rate limited by the Anthropic API. Wait a minute and try again." });
    } else if (err instanceof Anthropic.BadRequestError) {
      res.status(400).json({ error: `The API rejected the request: ${err.message}` });
    } else if (err instanceof Anthropic.APIError) {
      res.status(502).json({ error: `Anthropic API error (${err.status ?? "network"}). Try again shortly.` });
    } else if (err instanceof Error && /resolve authentication/i.test(err.message)) {
      // With no credentials configured the SDK throws a plain Error, not an APIError.
      res.status(401).json({ error: "No Anthropic credentials found. Set ANTHROPIC_API_KEY in .env." });
    } else {
      res.status(500).json({ error: "Something went wrong while analysing the video." });
    }
  }
});

// In production, serve the built front end from the same port.
const distDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../dist");
if (fs.existsSync(distDir)) {
  app.use(express.static(distDir));
  app.get("/{*splat}", (_req, res) => res.sendFile(path.join(distDir, "index.html")));
}

app.listen(PORT, () => {
  console.log(`Shuttle Coach API listening on http://localhost:${PORT} (model: ${MODEL})`);
  if (fs.existsSync(distDir)) console.log(`Serving the app from ${distDir}`);
});
