# Shuttle Coach

Upload a short badminton clip and get coaching feedback: scores, prioritised fixes tied to specific frames, and drills.

## How it works

1. **In the browser:** [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) tracks 33 body landmarks through the clip (sampled at up to 15 fps). From those, the app computes joint angles, stance width, trunk lean, hip-shoulder rotation, racket-arm speed and body movement. Swings are detected as peaks in racket-arm wrist speed.
2. **Key frames:** up to 10 moments (swing contact and preparation, deepest lunge, plus overview frames) are captured with the skeleton drawn on top.
3. **On the server:** the frames and measurements go to Claude (`claude-opus-5-5`), which returns a structured report validated against [shared/feedback.ts](shared/feedback.ts).
4. **The report** shows the video with a skeleton overlay, a timeline of arm speed and knee bend, and the feedback. Click any "Frame N" chip to jump to that moment.

The video itself never leaves your machine. Only the still frames and numbers are sent.

## Setup

Requires Node 20+.

```bash
npm install
cp .env.example .env    # then add your ANTHROPIC_API_KEY
```

## Run

```bash
# Development (hot reload): UI on http://localhost:5173, API on :3001
npm run dev

# Production: build once, then serve UI + API on http://localhost:3001
npm run build
npm start
```

Optional `.env` settings: `CLAUDE_MODEL`, `CLAUDE_EFFORT` (`low` to `max`, default `high`), `PORT`.

## Filming tips

- Full body in frame, feet included, from the side or from behind the baseline.
- One player clearly visible.
- Steady camera, good light, 5-30 seconds.

## Project layout

| Path | What it does |
|---|---|
| [src/lib/pose.ts](src/lib/pose.ts) | Loads MediaPipe and tracks pose frame by frame |
| [src/lib/metrics.ts](src/lib/metrics.ts) | Biomechanics, swing detection, clip summary |
| [src/lib/keyframes.ts](src/lib/keyframes.ts) | Picks and captures the key frames |
| [src/lib/analyze.ts](src/lib/analyze.ts) | Runs the whole pipeline and calls the API |
| [server/coach.ts](server/coach.ts) | The Claude request (structured output, refusal fallback) |
| [server/prompt.ts](server/prompt.ts) | Coaching system prompt and how measurements are described |

## Notes

- The pose model (~9 MB) is downloaded from Google's CDN on first use, then cached by the browser.
- Measurements come from one camera, so angles can be off by 10-15°. The coach is told this and weighs what it sees in the frames above the numbers.
- An analysis costs roughly 10-20k input tokens (mostly images) plus the report.
