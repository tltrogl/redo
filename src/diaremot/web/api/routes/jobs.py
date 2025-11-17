"""Job management routes."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from diaremot.pipeline.config import build_pipeline_config
from diaremot.web.api.models import JobCancelRequest, JobCreateRequest, JobListResponse, JobResponse
from diaremot.web.api.services import get_job_queue, get_storage
from diaremot.web.config_schema import generate_config_schema

router = APIRouter(prefix="/jobs", tags=["Jobs"])
logger = logging.getLogger(__name__)


@router.post("", response_model=JobResponse)
async def create_job(request: JobCreateRequest):
    """Create a new processing job.

    Args:
        request: Job creation request with file_id and configuration.

    Returns:
        Created job information.
    """
    storage = get_storage()
    job_queue = get_job_queue()

    # Build configuration
    config_overrides = request.config.copy()

    # Apply preset if specified
    if request.preset:
        schema = generate_config_schema()
        preset_data = schema["presets"].get(request.preset)
        if preset_data:
            # Merge preset overrides with request config (request takes precedence)
            for key, value in preset_data["overrides"].items():
                if key not in config_overrides:
                    config_overrides[key] = value
        else:
            raise HTTPException(status_code=400, detail=f"Unknown preset: {request.preset}")

    # Validate configuration
    try:
        config = build_pipeline_config(config_overrides)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {e}")

    # Infer file_id from filename if it looks like a UUID
    # This allows passing file_id as filename for convenience
    file_id = request.filename
    if len(file_id) == 36 and file_id.count("-") == 4:  # UUID format
        # Get actual filename from storage metadata
        file_path = storage.get_upload_path(file_id)
        if not file_path:
            raise HTTPException(status_code=404, detail=f"File {file_id} not found")
        actual_filename = file_path.name
    else:
        # filename is the actual filename, need to find the file_id
        # For now, treat filename as file_id
        file_path = storage.get_upload_path(request.filename)
        if not file_path:
            raise HTTPException(status_code=404, detail=f"File {request.filename} not found")
        actual_filename = request.filename

    # Get output directory
    job = job_queue.create_job(
        filename=actual_filename,
        input_path=file_path,
        output_dir=storage.get_output_dir("pending"),  # Will be updated when job starts
        config=config,
    )

    # Update output directory to use job_id
    job.output_dir = storage.get_output_dir(job.job_id)

    return JobResponse(**job.to_dict())


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Get job status and information."""
    job_queue = get_job_queue()
    job = job_queue.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobResponse(**job.to_dict())


@router.get("", response_model=JobListResponse)
async def list_jobs(limit: int = 100, offset: int = 0):
    """List all jobs."""
    job_queue = get_job_queue()
    jobs = job_queue.list_jobs(limit=limit, offset=offset)

    return JobListResponse(
        jobs=[JobResponse(**job.to_dict()) for job in jobs],
        total=len(jobs),
    )


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str, request: JobCancelRequest | None = None):
    """Cancel a running job."""
    job_queue = get_job_queue()
    success = await job_queue.cancel_job(job_id)

    if not success:
        raise HTTPException(
            status_code=400, detail=f"Job {job_id} cannot be cancelled (not found or already completed)"
        )

    return {"success": True, "message": "Job cancelled"}


@router.get("/{job_id}/results")
async def list_results(job_id: str):
    """List all result files for a job."""
    storage = get_storage()
    files = storage.list_result_files(job_id)

    return {"job_id": job_id, "files": files}


@router.get("/{job_id}/results/{filename}")
async def download_result(job_id: str, filename: str):
    """Download a specific result file."""
    storage = get_storage()
    file_path = storage.get_result_file(job_id, filename)

    if not file_path:
        raise HTTPException(status_code=404, detail=f"Result file {filename} not found for job {job_id}")

    return FileResponse(file_path, filename=filename)


@router.delete("/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and all its outputs."""
    job_queue = get_job_queue()
    storage = get_storage()

    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Delete outputs
    storage.cleanup_job_outputs(job_id)

    # Note: We keep the job record for history
    # Could add a flag to mark as deleted if needed

    return {"success": True, "message": "Job outputs deleted"}
