"""Health check routes."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter

from diaremot.web.api.models import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_dir = Path(os.getenv("DIAREMOT_MODEL_DIR", "models"))
    models_available = model_dir.exists() and any(model_dir.iterdir())

    status = "healthy" if models_available else "degraded"

    return HealthResponse(
        status=status,
        version="2.2.0",
        models_available=models_available,
        model_dir=str(model_dir),
    )


@router.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "DiaRemot Web API",
        "version": "2.2.0",
        "endpoints": {
            "health": "GET /health",
            "config_schema": "GET /api/config/schema",
            "presets": "GET /api/config/presets",
            "upload": "POST /api/files/upload",
            "create_job": "POST /api/jobs",
            "job_status": "GET /api/jobs/{job_id}",
            "job_progress": "WS /api/jobs/{job_id}/progress",
        },
    }
