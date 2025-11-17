"""Pydantic models for API request/response validation."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# Enums
class JobStatus(str, Enum):
    """Job processing status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStage(str, Enum):
    """Pipeline processing stages."""

    DEPENDENCY_CHECK = "dependency_check"
    PREPROCESS = "preprocess"
    BACKGROUND_SED = "background_sed"
    DIARIZE = "diarize"
    TRANSCRIBE = "transcribe"
    PARALINGUISTICS = "paralinguistics"
    AFFECT_AND_ASSEMBLE = "affect_and_assemble"
    OVERLAP_INTERRUPTIONS = "overlap_interruptions"
    CONVERSATION_ANALYSIS = "conversation_analysis"
    SPEAKER_ROLLUPS = "speaker_rollups"
    OUTPUTS = "outputs"


# Progress tracking
class JobProgress(BaseModel):
    """Job progress information."""

    stage: JobStage | None = None
    stage_index: int = 0
    stage_progress: float = 0.0
    overall_progress: float = 0.0
    message: str = ""


# Request models
class JobCreateRequest(BaseModel):
    """Request to create a new processing job."""

    filename: str = Field(..., description="Uploaded file ID or filename")
    config: dict[str, Any] = Field(default_factory=dict, description="Configuration overrides")
    preset: str | None = Field(None, description="Preset name to apply (optional)")


class JobCancelRequest(BaseModel):
    """Request to cancel a running job."""

    job_id: str = Field(..., description="Job ID to cancel")


class ConfigValidateRequest(BaseModel):
    """Request to validate a configuration."""

    config: dict[str, Any] = Field(..., description="Configuration to validate")


# Response models
class FileUploadResponse(BaseModel):
    """Response for file upload."""

    file_id: str = Field(..., description="Unique file identifier")
    filename: str = Field(..., description="Original filename")
    size: int = Field(..., description="File size in bytes")
    duration: float | None = Field(None, description="Audio duration in seconds")
    sample_rate: int | None = Field(None, description="Sample rate in Hz")
    channels: int | None = Field(None, description="Number of audio channels")


class JobResponse(BaseModel):
    """Response for job information."""

    job_id: str
    status: JobStatus
    filename: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    progress: JobProgress
    error: str | None = None
    config: dict[str, Any]
    results: dict[str, Any] | None = None


class JobListResponse(BaseModel):
    """Response for job list."""

    jobs: list[JobResponse]
    total: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status: healthy, degraded, unhealthy")
    version: str = Field(..., description="API version")
    models_available: bool = Field(..., description="Whether required models are available")
    model_dir: str = Field(..., description="Model directory path")


class ConfigParameter(BaseModel):
    """Configuration parameter metadata."""

    name: str
    type: str
    default: Any
    label: str
    description: str
    group: str
    ui_type: str
    advanced: bool
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list[str] | None = None
    unit: str | None = None
    nullable: bool = False
    placeholder: str | None = None


class ConfigGroup(BaseModel):
    """Configuration group metadata."""

    label: str
    description: str
    icon: str
    order: int


class ConfigSchemaResponse(BaseModel):
    """Complete configuration schema."""

    parameters: dict[str, ConfigParameter]
    groups: dict[str, ConfigGroup]
    defaults: dict[str, Any]
    presets: dict[str, Any]


class PresetResponse(BaseModel):
    """Configuration preset."""

    name: str = Field(..., description="Preset identifier")
    label: str = Field(..., description="Display label")
    description: str = Field(..., description="Preset description")
    overrides: dict[str, Any] = Field(..., description="Configuration overrides")
