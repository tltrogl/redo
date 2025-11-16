"""Main FastAPI application."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from diaremot.web.api.routes import config, files, health, jobs
from diaremot.web.api.services import initialize_job_queue, shutdown_job_queue
from diaremot.web.api.websocket import progress

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="DiaRemot Web API",
    description="Speech intelligence pipeline with comprehensive audio analysis",
    version="2.2.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(config.router, prefix="/api")
app.include_router(files.router, prefix="/api")
app.include_router(jobs.router, prefix="/api")
app.include_router(progress.router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting DiaRemot Web API...")

    # Initialize job queue
    await initialize_job_queue()

    logger.info("DiaRemot Web API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down DiaRemot Web API...")

    # Shutdown job queue
    await shutdown_job_queue()

    logger.info("DiaRemot Web API shutdown complete")


if __name__ == "__main__":
    import uvicorn

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
