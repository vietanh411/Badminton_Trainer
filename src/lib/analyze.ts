import type { AnalyzeResponse, Feedback } from "../../shared/feedback";
import type { FrameMetrics, Hand, PlayerContext, SessionSummary } from "../../shared/types";
import { captureKeyFrames, selectKeyFrames, type CapturedFrame } from "./keyframes";
import { computeFrameMetrics, detectHand, detectStrokes, smoothSamples, summarise, type DetectedStroke } from "./metrics";
import { chooseSampleFps, loadPoseLandmarker, trackPose, type PoseSample } from "./pose";

export type Stage = "loading" | "tracking" | "capturing" | "coaching";

export interface Progress {
  stage: Stage;
  /** 0-1 within the current stage; null when the stage has no measurable progress. */
  fraction: number | null;
  detail?: string;
}

export interface AnalysisResult {
  feedback: Feedback;
  model: string;
  frames: CapturedFrame[];
  samples: PoseSample[];
  metrics: FrameMetrics[];
  strokes: DetectedStroke[];
  summary: SessionSummary;
  hand: Hand;
}

export const MAX_DURATION_SEC = 120;

export async function loadVideo(url: string): Promise<HTMLVideoElement> {
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  await new Promise<void>((resolve, reject) => {
    video.onloadeddata = () => resolve();
    video.onerror = () =>
      reject(new Error("Your browser can't play this video. Try an MP4 (H.264) file, or re-export it from your phone."));
    video.src = url;
  });

  // WebM files from MediaRecorder / screen recorders often report an infinite duration until
  // the browser has scanned to the end. Seeking far past the end forces it to work it out.
  if (video.duration === Infinity) {
    await new Promise<void>((resolve) => {
      const done = () => {
        if (video.duration === Infinity) return;
        video.removeEventListener("durationchange", done);
        resolve();
      };
      video.addEventListener("durationchange", done);
      video.currentTime = 1e7;
      setTimeout(resolve, 5000);
    });
    video.currentTime = 0;
  }
  return video;
}

export async function runAnalysis(
  videoUrl: string,
  context: PlayerContext,
  onProgress: (p: Progress) => void,
  signal: AbortSignal,
): Promise<AnalysisResult> {
  onProgress({ stage: "loading", fraction: null, detail: "Loading the pose model" });
  const [video] = await Promise.all([loadVideo(videoUrl), loadPoseLandmarker()]);

  if (!Number.isFinite(video.duration) || video.duration <= 0) {
    throw new Error("Couldn't read the video's length. Try re-exporting it as MP4.");
  }
  if (video.duration > MAX_DURATION_SEC) {
    throw new Error(`Please trim the clip to under ${MAX_DURATION_SEC / 60} minutes. Short clips (5-30 s) work best.`);
  }

  const fps = chooseSampleFps(video.duration);
  const raw = await trackPose(
    video,
    fps,
    (fraction) => onProgress({ stage: "tracking", fraction, detail: `Tracking body position at ${fps} fps` }),
    signal,
  );

  const coverage = raw.filter((s) => s.image).length / raw.length;
  if (coverage < 0.2) {
    throw new Error(
      "Couldn't find a player in most of the video. Make sure one person is clearly visible, ideally full-body.",
    );
  }

  const samples = smoothSamples(raw);
  const hand = context.hand === "auto" ? detectHand(samples) : context.hand;
  const { videoWidth: width, videoHeight: height } = video;
  const metrics = computeFrameMetrics(samples, hand, width, height);
  const strokes = detectStrokes(samples, metrics);
  const summary = summarise(samples, metrics, strokes, {
    durationSec: video.duration,
    width,
    height,
    fps,
    hand,
    handAutoDetected: context.hand === "auto",
  });

  const picks = selectKeyFrames(samples, metrics, strokes);
  const frames = await captureKeyFrames(video, samples, metrics, picks, hand, (fraction) =>
    onProgress({ stage: "capturing", fraction, detail: `Capturing ${picks.length} key frames` }),
  );
  video.removeAttribute("src");
  video.load();

  onProgress({ stage: "coaching", fraction: null, detail: `Coach is reviewing ${frames.length} frames` });
  const res = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      context,
      summary,
      keyFrames: frames.map(({ dataUrl: _dataUrl, ...frame }) => frame),
    }),
    signal,
  });

  const body = await res.json().catch(() => null);
  if (!res.ok) {
    throw new Error(body?.error ?? `The analysis server returned an error (${res.status}).`);
  }
  const { feedback, model } = body as AnalyzeResponse;

  return { feedback, model, frames, samples, metrics, strokes, summary, hand };
}
