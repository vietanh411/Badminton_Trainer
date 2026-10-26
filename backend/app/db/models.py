"""Relational data model.

Small, queryable facts live in the database; bulky per-frame data (tracks, pose landmarks,
positions) lives in Parquet / NPZ artifacts on disk, referenced by path. Videos are never stored
as BLOBs.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class User(Base):
    """Optional: the local MVP is single-user; kept so a multi-user deployment has a place to start."""

    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    display_name: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Video(Base):
    __tablename__ = "videos"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    original_filename: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str] = mapped_column(String(500))  # relative to data_dir
    proxy_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    thumbnail_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    truth_path: Mapped[str | None] = mapped_column(String(500), nullable=True)  # synthetic only
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int] = mapped_column(Integer)
    container: Mapped[str] = mapped_column(String(16))
    codec: Mapped[str] = mapped_column(String(16))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    fps: Mapped[float] = mapped_column(Float)
    frame_count: Mapped[int] = mapped_column(Integer)
    duration_s: Mapped[float] = mapped_column(Float)
    rotation: Mapped[int] = mapped_column(Integer, default=0)
    browser_playable: Mapped[bool] = mapped_column(Boolean, default=True)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    sessions: Mapped[list[VideoSession]] = relationship(back_populates="video")


class VideoSession(Base):
    __tablename__ = "video_sessions"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(200))
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), default="draft"
    )  # draft|queued|processing|completed|failed|cancelled
    data_source: Mapped[str] = mapped_column(String(20), default="upload")  # upload | synthetic
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    settings: Mapped[dict] = mapped_column(JSON, default=dict)
    player_selection: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    corrections: Mapped[list] = mapped_column(JSON, default=list)
    summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    results_dir: Mapped[str | None] = mapped_column(String(500), nullable=True)
    latest_job_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    video: Mapped[Video] = relationship(back_populates="sessions")
    calibrations: Mapped[list[CourtCalibration]] = relationship(back_populates="session", cascade="all, delete-orphan")
    jobs: Mapped[list[AnalysisJob]] = relationship(back_populates="session", cascade="all, delete-orphan")
    strokes: Mapped[list[StrokeEvent]] = relationship(back_populates="session", cascade="all, delete-orphan")


class CourtCalibration(Base):
    __tablename__ = "court_calibrations"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("video_sessions.id", ondelete="CASCADE"), index=True)
    method: Mapped[str] = mapped_column(String(20))  # manual | auto
    frame_time_s: Mapped[float] = mapped_column(Float, default=0.0)
    image_width: Mapped[int] = mapped_column(Integer)
    image_height: Mapped[int] = mapped_column(Integer)
    point_ids: Mapped[list] = mapped_column(JSON)
    image_points: Mapped[list] = mapped_column(JSON)
    homography: Mapped[list] = mapped_column(JSON)
    inliers: Mapped[list] = mapped_column(JSON)
    errors_m: Mapped[list] = mapped_column(JSON)
    errors_px: Mapped[list] = mapped_column(JSON)
    mean_error_m: Mapped[float] = mapped_column(Float)
    max_error_m: Mapped[float] = mapped_column(Float)
    mean_error_px: Mapped[float] = mapped_column(Float)
    condition_number: Mapped[float] = mapped_column(Float)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    session: Mapped[VideoSession] = relationship(back_populates="calibrations")


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("video_sessions.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(20), default="QUEUED", index=True)
    stage: Mapped[str | None] = mapped_column(String(32), nullable=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    settings: Mapped[dict] = mapped_column(JSON, default=dict)
    timings: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    session: Mapped[VideoSession] = relationship(back_populates="jobs")


class PlayerTrack(Base):
    """Summary of one tracker identity in one analysis run (per-frame boxes are in tracks.parquet)."""

    __tablename__ = "player_tracks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("video_sessions.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("analysis_jobs.id", ondelete="CASCADE"))
    track_id: Mapped[int] = mapped_column(Integer)
    first_t: Mapped[float] = mapped_column(Float)
    last_t: Mapped[float] = mapped_column(Float)
    n_frames: Mapped[int] = mapped_column(Integer)
    mean_score: Mapped[float] = mapped_column(Float)
    is_target: Mapped[bool] = mapped_column(Boolean, default=False)


class PoseLandmarks(Base):
    __tablename__ = "pose_landmarks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("video_sessions.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("analysis_jobs.id", ondelete="CASCADE"))
    artifact_path: Mapped[str] = mapped_column(String(500))
    model_name: Mapped[str] = mapped_column(String(100))
    frames_with_pose: Mapped[int] = mapped_column(Integer)


class CourtPositions(Base):
    __tablename__ = "court_positions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("video_sessions.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("analysis_jobs.id", ondelete="CASCADE"))
    artifact_path: Mapped[str] = mapped_column(String(500))
    n_samples: Mapped[int] = mapped_column(Integer)
    n_with_position: Mapped[int] = mapped_column(Integer)
    n_interpolated: Mapped[int] = mapped_column(Integer)


class MovementMetrics(Base):
    __tablename__ = "movement_metrics"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("video_sessions.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("analysis_jobs.id", ondelete="CASCADE"))
    pipeline_version: Mapped[str] = mapped_column(String(20))
    metrics: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class StrokeEvent(Base):
    __tablename__ = "stroke_events"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("video_sessions.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("analysis_jobs.id", ondelete="SET NULL"), nullable=True)
    t: Mapped[float] = mapped_column(Float, index=True)
    t_start: Mapped[float | None] = mapped_column(Float, nullable=True)
    t_end: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(20), default="detected")  # detected | manual
    review_status: Mapped[str] = mapped_column(String(20), default="candidate")  # candidate | confirmed | rejected
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    shot_type: Mapped[str | None] = mapped_column(String(40), nullable=True)
    shot_type_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    stroke_side: Mapped[str | None] = mapped_column(String(20), nullable=True)
    stroke_side_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    height_class: Mapped[str | None] = mapped_column(String(20), nullable=True)
    classifier: Mapped[str | None] = mapped_column(String(60), nullable=True)
    signals: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    label_shot_type: Mapped[str | None] = mapped_column(String(40), nullable=True)  # human annotation
    label_stroke_side: Mapped[str | None] = mapped_column(String(20), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    session: Mapped[VideoSession] = relationship(back_populates="strokes")


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    task: Mapped[str] = mapped_column(String(40))  # stroke_shot_type | stroke_side
    version: Mapped[str] = mapped_column(String(80), unique=True)
    path: Mapped[str] = mapped_column(String(500))
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    card: Mapped[dict] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ExportArtifact(Base):
    __tablename__ = "export_artifacts"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("video_sessions.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(40))
    path: Mapped[str] = mapped_column(String(500))
    size_bytes: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
