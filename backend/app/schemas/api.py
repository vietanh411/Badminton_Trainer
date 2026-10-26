"""Request / response models (they also drive the OpenAPI documentation at /docs)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class VideoOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    original_filename: str
    size_bytes: int
    container: str
    codec: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    rotation: int
    browser_playable: bool
    has_proxy: bool = False
    is_synthetic: bool
    created_at: datetime


class CalibrationPoint(BaseModel):
    id: str = Field(description="court keypoint id, e.g. 'near_baseline__left_doubles' (see GET /api/court)")
    x: float = Field(description="image x in pixels of the full-resolution video frame")
    y: float


class CalibrationIn(BaseModel):
    points: list[CalibrationPoint] = Field(min_length=4)
    frame_time_s: float = 0.0
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    dry_run: bool = Field(default=False, description="validate and report errors without saving")

    @field_validator("points")
    @classmethod
    def unique_ids(cls, v: list[CalibrationPoint]) -> list[CalibrationPoint]:
        ids = [p.id for p in v]
        if len(set(ids)) != len(ids):
            raise ValueError("each court landmark may be used only once")
        return v


class CalibrationOut(BaseModel):
    id: str | None = None
    method: str
    frame_time_s: float
    image_width: int
    image_height: int
    point_ids: list[str]
    image_points: list[list[float]]
    homography: list[list[float]]
    inliers: list[bool]
    errors_m: list[float]
    errors_px: list[float]
    mean_error_m: float
    max_error_m: float
    mean_error_px: float
    condition_number: float
    warnings: list[str]
    quality: Literal["good", "fair", "poor"]
    saved: bool = False


class PlayerSelectionIn(BaseModel):
    t: float = Field(ge=0)
    bbox: list[float] = Field(min_length=4, max_length=4, description="x1, y1, x2, y2 in full-resolution pixels")

    @field_validator("bbox")
    @classmethod
    def valid_box(cls, v: list[float]) -> list[float]:
        if v[2] <= v[0] or v[3] <= v[1]:
            raise ValueError("bbox must be x1 < x2 and y1 < y2")
        return v


class CorrectionIn(BaseModel):
    t: float = Field(ge=0)
    track_id: int | None = Field(description="tracker id that is the target from time t; null = target not visible")


class AnalyzeIn(BaseModel):
    performance_mode: Literal["fast", "balanced", "accurate"] | None = None
    detector: Literal["auto", "yolo", "mediapipe"] | None = None
    tracker: Literal["bytetrack", "botsort"] | None = None
    analysis_fps: float | None = Field(default=None, gt=0, le=60)
    max_width: int | None = Field(default=None, ge=320, le=3840)
    detect_strokes: bool | None = None
    trajectory: dict[str, Any] | None = None
    movement: dict[str, Any] | None = None
    heatmap: dict[str, Any] | None = None
    strokes: dict[str, Any] | None = None
    target: dict[str, Any] | None = None


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    session_id: str
    status: str
    stage: str | None
    progress: float
    message: str | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    timings: dict | None = None


class SessionPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    notes: str | None = Field(default=None, max_length=5000)


class StrokeIn(BaseModel):
    t: float = Field(ge=0)
    label_shot_type: str | None = None
    label_stroke_side: Literal["forehand", "backhand"] | None = None


class StrokePatch(BaseModel):
    review_status: Literal["candidate", "confirmed", "rejected"] | None = None
    t: float | None = Field(default=None, ge=0)
    label_shot_type: Literal["smash", "clear", "drop", "drive", "net_shot", "lift", "other"] | None = None
    label_stroke_side: Literal["forehand", "backhand"] | None = None
    clear_labels: bool = False
