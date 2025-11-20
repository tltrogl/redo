"""WebSocket endpoint for real-time job progress updates."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from diaremot.web.api.services import get_job_queue

router = APIRouter(tags=["WebSocket"])
logger = logging.getLogger(__name__)


@router.websocket("/jobs/{job_id}/progress")
async def job_progress_websocket(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job progress updates.

    Args:
        websocket: WebSocket connection.
        job_id: Job identifier.
    """
    await websocket.accept()
    logger.info("WebSocket connection established for job %s", job_id)

    job_queue = get_job_queue()
    job = job_queue.get_job(job_id)

    if not job:
        await websocket.send_json(
            {
                "type": "error",
                "message": f"Job {job_id} not found",
            }
        )
        await websocket.close()
        return

    # Subscribe to job progress
    progress_queue = job.subscribe_progress()

    try:
        # Send current progress immediately
        await websocket.send_json(
            {
                "type": "progress",
                "job_id": job_id,
                "progress": job.progress.model_dump(),
                "status": job.status.value,
            }
        )

        # Listen for progress updates
        while True:
            try:
                # Wait for progress update with timeout to check connection
                try:
                    progress_data = await asyncio.wait_for(progress_queue.get(), timeout=1.0)
                    await websocket.send_json(progress_data)

                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    try:
                        await websocket.send_json({"type": "heartbeat"})
                    except Exception:
                        # Connection lost
                        break

                # Check if job is complete
                if job.status.value in ("completed", "failed", "cancelled"):
                    # Send final status
                    await websocket.send_json(
                        {
                            "type": "complete",
                            "job_id": job_id,
                            "status": job.status.value,
                            "results": job.results,
                            "error": job.error,
                        }
                    )
                    break

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected for job %s", job_id)
                break

            except Exception as e:
                logger.exception("WebSocket error for job %s: %s", job_id, e)
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": str(e),
                    }
                )
                break

    finally:
        # Unsubscribe from progress updates
        job.unsubscribe_progress(progress_queue)
        logger.info("WebSocket connection closed for job %s", job_id)


@router.websocket("/jobs/subscribe")
async def jobs_subscribe_websocket(websocket: WebSocket):
    """WebSocket endpoint for subscribing to all job updates.

    Useful for a dashboard view showing all active jobs.
    """
    await websocket.accept()
    logger.info("Global jobs WebSocket connection established")

    job_queue = get_job_queue()

    try:
        # Send current jobs list
        jobs = job_queue.list_jobs(limit=100)
        await websocket.send_json(
            {
                "type": "jobs_list",
                "jobs": [job.to_dict() for job in jobs],
            }
        )

        # Keep connection alive and listen for client messages
        while True:
            try:
                # Wait for client message or send heartbeat
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                    message = json.loads(data)

                    if message.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})

                    elif message.get("type") == "refresh":
                        # Send updated jobs list
                        jobs = job_queue.list_jobs(limit=100)
                        await websocket.send_json(
                            {
                                "type": "jobs_list",
                                "jobs": [job.to_dict() for job in jobs],
                            }
                        )

                except asyncio.TimeoutError:
                    # Send heartbeat
                    await websocket.send_json({"type": "heartbeat"})

            except WebSocketDisconnect:
                logger.info("Global jobs WebSocket disconnected")
                break

            except Exception as e:
                logger.exception("Global jobs WebSocket error: %s", e)
                break

    finally:
        logger.info("Global jobs WebSocket connection closed")
