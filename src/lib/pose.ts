import { PoseLandmarker } from "@mediapipe/tasks-vision";
import wasmLoaderPath from "@mediapipe/tasks-vision/vision_wasm_internal.js?url";
import wasmBinaryPath from "@mediapipe/tasks-vision/vision_wasm_internal.wasm?url";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task";

export interface Pt {
  x: number;
  y: number;
  z: number;
  v: number;
}

export interface PoseSample {
  /** Seconds into the video. */
  t: number;
  /** Normalised image coordinates (0-1), 33 landmarks. */
  image: Pt[] | null;
  /** Metric 3D coordinates centred on the hips, 33 landmarks. */
  world: Pt[] | null;
}

let landmarker: Promise<PoseLandmarker> | null = null;

export function loadPoseLandmarker(): Promise<PoseLandmarker> {
  landmarker ??= (async () => {
    const fileset = { wasmLoaderPath, wasmBinaryPath };
    const options = (delegate: "GPU" | "CPU") => ({
      baseOptions: { modelAssetPath: MODEL_URL, delegate },
      runningMode: "VIDEO" as const,
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    try {
      return await PoseLandmarker.createFromOptions(fileset, options("GPU"));
    } catch (err) {
      console.warn("GPU pose model failed, falling back to CPU", err);
      return PoseLandmarker.createFromOptions(fileset, options("CPU"));
    }
  })();
  landmarker.catch(() => (landmarker = null));
  return landmarker;
}

export function seek(video: HTMLVideoElement, t: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const done = () => {
      video.removeEventListener("seeked", done);
      video.removeEventListener("error", fail);
      resolve();
    };
    const fail = () => {
      video.removeEventListener("seeked", done);
      video.removeEventListener("error", fail);
      reject(new Error("The video couldn't be decoded at this point."));
    };
    video.addEventListener("seeked", done);
    video.addEventListener("error", fail);
    video.currentTime = t;
  });
}

/** Pick a sampling rate that keeps processing time reasonable for longer clips. */
export function chooseSampleFps(durationSec: number): number {
  const MAX_SAMPLES = 450;
  return Math.max(4, Math.min(15, Math.floor(MAX_SAMPLES / Math.max(durationSec, 1))));
}

export async function trackPose(
  video: HTMLVideoElement,
  fps: number,
  onProgress: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<PoseSample[]> {
  const model = await loadPoseLandmarker();
  const samples: PoseSample[] = [];
  const step = 1 / fps;
  const total = Math.max(1, Math.floor(video.duration / step));
  let lastTs = -1;

  for (let i = 0; i < total; i++) {
    if (signal?.aborted) throw new DOMException("Cancelled", "AbortError");
    const t = Math.min(i * step, video.duration - 0.01);
    await seek(video, t);

    // MediaPipe needs strictly increasing timestamps in VIDEO mode.
    const ts = Math.max(Math.round(t * 1000), lastTs + 1);
    lastTs = ts;

    const result = model.detectForVideo(video, ts);
    const image = result.landmarks[0];
    const world = result.worldLandmarks[0];
    samples.push({
      t,
      image: image ? image.map((p) => ({ x: p.x, y: p.y, z: p.z, v: p.visibility ?? 1 })) : null,
      world: world ? world.map((p) => ({ x: p.x, y: p.y, z: p.z, v: p.visibility ?? 1 })) : null,
    });
    onProgress((i + 1) / total);
  }
  return samples;
}
