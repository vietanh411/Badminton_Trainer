import { describe, expect, it } from "vitest";
import { clipRange } from "./clip";
import { gridToRgba, heatColor, HEAT_STOPS } from "./colormap";
import { fmtBytes, fmtDuration, fmtNum, fmtPct, fmtSigned, fmtTime, titleCase } from "./format";
import { invert3, multiply, project, projectSegment, scaleMat, type Mat3 } from "./homography";
import { lowerBound, nearestIndex, windowBounds, windowIndex, windowsToLoad } from "./timeindex";

// A realistic image→court homography (from the synthetic baseline camera).
const H: Mat3 = [
  [0.0119, 0.0021, -5.21],
  [0.0003, -0.0151, 9.72],
  [0.00001, 0.0011, 0.32],
];

describe("homography", () => {
  it("inverts and round-trips points", () => {
    const Hi = invert3(H)!;
    const I = multiply(H, Hi);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) expect(I[i][j]).toBeCloseTo(i === j ? 1 : 0, 9);
    const court: [number, number] = [2.5, 4.0];
    const img = project(Hi, court)!;
    const back = project(H, img)!;
    expect(back[0]).toBeCloseTo(2.5, 9);
    expect(back[1]).toBeCloseTo(4.0, 9);
  });

  it("returns null for singular matrices and points at infinity", () => {
    expect(invert3([[1, 2, 3], [2, 4, 6], [0, 0, 1]])).toBeNull();
    expect(project([[1, 0, 0], [0, 1, 0], [1, 0, 0]], [0, 5])).toBeNull();
  });

  it("composes with scaling for resized frames", () => {
    const S = scaleMat(2, 2); // display px → full-res px
    const p = project(multiply(H, S), [100, 50])!;
    const q = project(H, [200, 100])!;
    expect(p[0]).toBeCloseTo(q[0], 9);
    expect(p[1]).toBeCloseTo(q[1], 9);
  });

  it("splits segments that cross behind the camera", () => {
    // w = y: points with y < 0 are behind the camera.
    const P: Mat3 = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 1, 0],
    ];
    const runs = projectSegment(P, [0, -1], [0, 1], 10);
    expect(runs.length).toBe(1);
    expect(runs[0].every(([, y]) => Number.isFinite(y))).toBe(true);
  });
});

describe("colormap", () => {
  it("interpolates between stops and clamps", () => {
    expect(heatColor(-1)).toEqual(HEAT_STOPS[0].c);
    expect(heatColor(2)).toEqual(HEAT_STOPS[4].c);
    expect(heatColor(0.45)).toEqual(HEAT_STOPS[2].c);
    const mid = heatColor(0.3);
    expect(mid[0]).toBeGreaterThan(13);
    expect(mid[0]).toBeLessThan(34);
    expect(heatColor(Number.NaN)[3]).toBe(0);
  });

  it("maps grids to RGBA with vertical flip", () => {
    const px = gridToRgba([[0, 0], [0, 1]], 1);
    expect(px.length).toBe(16);
    // After flipping, the top-left pixel comes from the last grid row: [0, 1] → transparent then bright.
    expect(px[3]).toBe(0);
    expect(px[7]).toBe(255);
    expect(gridToRgba([[1]], 1, 0.5)[3]).toBe(128);
  });
});

describe("time index", () => {
  const frames = [0, 0.1, 0.2, 0.4, 0.5].map((t) => ({ t }));
  it("binary searches", () => {
    expect(lowerBound(frames, 0.2, (f) => f.t)).toBe(2);
    expect(lowerBound(frames, 0.3, (f) => f.t)).toBe(3);
    expect(lowerBound(frames, 9, (f) => f.t)).toBe(5);
    expect(nearestIndex(frames, 0.31, (f) => f.t)).toBe(3);
    expect(nearestIndex(frames, 0.29, (f) => f.t)).toBe(2);
    expect(nearestIndex(frames, 3, (f) => f.t, 0.2)).toBe(-1);
    expect(nearestIndex([], 1, (f: { t: number }) => f.t)).toBe(-1);
  });

  it("computes aligned windows and prefetch", () => {
    expect(windowIndex(0)).toBe(0);
    expect(windowIndex(45)).toBe(1);
    expect(windowBounds(2)).toEqual({ start: 60, end: 90 });
    expect(windowsToLoad(5, 100)).toEqual([0]);
    expect(windowsToLoad(25, 100)).toEqual([0, 1]);
    expect(windowsToLoad(25, 29)).toEqual([0]);
  });
});

describe("format", () => {
  it("formats numbers and handles missing values", () => {
    expect(fmtNum(3.14159, 2, "m")).toBe("3.14 m");
    expect(fmtNum(null)).toBe("—");
    expect(fmtNum(Number.NaN)).toBe("—");
    expect(fmtPct(0.256)).toBe("26%");
    expect(fmtDuration(125.7)).toBe("2:05");
    expect(fmtTime(65.25)).toBe("1:05.3");
    expect(fmtBytes(1536)).toBe("1.5 KB");
    expect(fmtSigned(0.5)).toBe("+0.50");
    expect(fmtSigned(-0.5, 1)).toBe("-0.5");
    expect(titleCase("front_left")).toBe("Front Left");
  });
});

describe("annotated clip range", () => {
  it("clamps to 60 s, the video length and a positive span", () => {
    expect(clipRange(10, 30, 100)).toEqual([10, 30]);
    expect(clipRange(10, 200, 300)).toEqual([10, 70]);
    expect(clipRange(95, 115, 100)).toEqual([95, 100]);
    expect(clipRange(-5, -1, 100)).toEqual([0, 0.5]);
    expect(clipRange(120, 130, 100)).toEqual([99.5, 100]);
  });
});

import { boxFromDrag, clientToImage, hitBox, nextPersonId, normaliseDoc, toSavedDoc } from "./annotation";

describe("annotation helpers", () => {
  it("builds clamped boxes from drags in any direction and rejects tiny ones", () => {
    expect(boxFromDrag([50, 80], [10, 20], 100, 100)).toEqual([10, 20, 50, 80]);
    expect(boxFromDrag([-5, -5], [200, 300], 100, 100)).toEqual([0, 0, 100, 100]);
    expect(boxFromDrag([10, 10], [12, 40], 100, 100)).toBeNull();
  });
  it("maps screen points to image pixels", () => {
    expect(clientToImage(150, 100, { left: 100, top: 50, width: 200, height: 100 }, 1280, 720)).toEqual([320, 360]);
  });
  it("hit-tests the smallest containing box", () => {
    const people = [
      { id: "P1", bbox: [0, 0, 100, 100] as [number, number, number, number], ground_contact: null },
      { id: "P2", bbox: [40, 40, 60, 60] as [number, number, number, number], ground_contact: null },
    ];
    expect(hitBox(people, [50, 50])).toBe(1);
    expect(hitBox(people, [5, 5])).toBe(0);
    expect(hitBox(people, [500, 5])).toBe(-1);
  });
  it("only saves reviewed frames as ground truth", () => {
    const doc = normaliseDoc({
      format: "badmintonvision-annotations/1",
      source: "pre-annotation",
      warning: "machine generated",
      target_id: "T1",
      frames: [
        { t: 1, people: [{ id: "T1", bbox: [0, 0, 10, 20], ground_contact: null }] },
        { t: 0.5, people: [] },
      ],
    });
    expect(doc.frames.map((f) => f.t)).toEqual([0.5, 1]);
    expect(doc.frames.every((f) => f.reviewed === false)).toBe(true);
    expect(nextPersonId(doc)).toBe("P1");
    doc.frames[1].reviewed = true;
    const saved = toSavedDoc(doc);
    expect(saved.frames.map((f) => f.t)).toEqual([1]);
    expect(saved.unreviewed_frames?.map((f) => f.t)).toEqual([0.5]);
    expect(saved.source).toBe("manual");
    expect("warning" in saved).toBe(false);
  });
});
