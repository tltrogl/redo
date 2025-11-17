"""Job queue service for background pipeline processing."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from diaremot.web.api.models import JobProgress, JobStage, JobStatus

logger = logging.getLogger(__name__)


class Job:
    """Represents a single processing job."""

    def __init__(
        self,
        job_id: str,
        filename: str,
        input_path: Path,
        output_dir: Path,
        config: dict[str, Any],
    ):
        self.job_id = job_id
        self.filename = filename
        self.input_path = input_path
        self.output_dir = output_dir
        self.config = config

        self.status = JobStatus.PENDING
        self.created_at = datetime.now(timezone.utc)
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None

        self.progress = JobProgress()
        self.error: str | None = None
        self.results: dict[str, Any] | None = None

        self._progress_subscribers: list[asyncio.Queue] = []

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary."""
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress.model_dump(),
            "error": self.error,
            "config": self.config,
            "results": self.results,
        }

    def subscribe_progress(self) -> asyncio.Queue:
        """Subscribe to progress updates.

        Returns:
            Queue that will receive progress updates.
        """
        queue: asyncio.Queue = asyncio.Queue()
        self._progress_subscribers.append(queue)
        return queue

    def unsubscribe_progress(self, queue: asyncio.Queue) -> None:
        """Unsubscribe from progress updates."""
        if queue in self._progress_subscribers:
            self._progress_subscribers.remove(queue)

    async def update_progress(
        self,
        stage: JobStage | None = None,
        stage_index: int | None = None,
        stage_progress: float | None = None,
        message: str | None = None,
    ) -> None:
        """Update job progress and notify subscribers."""
        if stage is not None:
            self.progress.stage = stage
        if stage_index is not None:
            self.progress.stage_index = stage_index
        if stage_progress is not None:
            self.progress.stage_progress = stage_progress
        if message is not None:
            self.progress.message = message

        # Calculate overall progress (11 stages total)
        base_progress = (self.progress.stage_index / 11) * 100
        stage_contribution = (self.progress.stage_progress / 11)
        self.progress.overall_progress = base_progress + stage_contribution

        # Notify subscribers
        progress_data = {
            "type": "progress",
            "job_id": self.job_id,
            "progress": self.progress.model_dump(),
        }

        for queue in self._progress_subscribers[:]:  # Copy to avoid modification during iteration
            try:
                queue.put_nowait(progress_data)
            except asyncio.QueueFull:
                logger.warning("Progress queue full for job %s, removing subscriber", self.job_id)
                self._progress_subscribers.remove(queue)


class JobQueue:
    """Manages background job processing."""

    def __init__(self, jobs_dir: Path):
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

        self._jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._running = False

        # Load existing jobs from disk
        self._load_jobs()

    def _load_jobs(self) -> None:
        """Load existing jobs from disk."""
        for job_file in self.jobs_dir.glob("*.json"):
            try:
                with open(job_file) as f:
                    data = json.load(f)

                job_id = data["job_id"]
                job = Job(
                    job_id=job_id,
                    filename=data["filename"],
                    input_path=Path(data["input_path"]),
                    output_dir=Path(data["output_dir"]),
                    config=data.get("config", {}),
                )

                job.status = JobStatus(data["status"])
                job.created_at = datetime.fromisoformat(data["created_at"])
                if data.get("started_at"):
                    job.started_at = datetime.fromisoformat(data["started_at"])
                if data.get("completed_at"):
                    job.completed_at = datetime.fromisoformat(data["completed_at"])
                job.error = data.get("error")
                job.results = data.get("results")

                if "progress" in data:
                    job.progress = JobProgress(**data["progress"])

                self._jobs[job_id] = job

                logger.info("Loaded job %s with status %s", job_id, job.status)

                # Re-queue pending jobs
                if job.status == JobStatus.PENDING:
                    self._queue.put_nowait(job)

            except Exception as e:
                logger.error("Failed to load job from %s: %s", job_file, e)

    def _save_job(self, job: Job) -> None:
        """Save job state to disk."""
        job_file = self.jobs_dir / f"{job.job_id}.json"
        try:
            # Don't save input_path in the JSON as it may be temporary
            data = job.to_dict()
            data["input_path"] = str(job.input_path)
            data["output_dir"] = str(job.output_dir)

            with open(job_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Failed to save job %s: %s", job.job_id, e)

    def create_job(
        self,
        filename: str,
        input_path: Path,
        output_dir: Path,
        config: dict[str, Any],
    ) -> Job:
        """Create a new job and add it to the queue.

        Args:
            filename: Original filename.
            input_path: Path to uploaded audio file.
            output_dir: Output directory for results.
            config: Pipeline configuration.

        Returns:
            Created job.
        """
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            filename=filename,
            input_path=input_path,
            output_dir=output_dir,
            config=config,
        )

        self._jobs[job_id] = job
        self._save_job(job)
        self._queue.put_nowait(job)

        logger.info("Created job %s for file %s", job_id, filename)
        return job

    def get_job(self, job_id: str) -> Job | None:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 100, offset: int = 0) -> list[Job]:
        """List all jobs."""
        jobs = sorted(
            self._jobs.values(),
            key=lambda j: j.created_at,
            reverse=True,
        )
        return jobs[offset : offset + limit]

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job.

        Args:
            job_id: Job identifier.

        Returns:
            True if job was cancelled, False if not found or already completed.
        """
        job = self._jobs.get(job_id)
        if not job:
            return False

        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return False

        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now(timezone.utc)
        job.error = "Cancelled by user"
        self._save_job(job)

        # Notify subscribers
        await job.update_progress(message="Job cancelled")

        logger.info("Cancelled job %s", job_id)
        return True

    async def start(self) -> None:
        """Start the job queue worker."""
        if self._running:
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("Job queue worker started")

    async def stop(self) -> None:
        """Stop the job queue worker."""
        if not self._running:
            return

        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        logger.info("Job queue worker stopped")

    async def _worker(self) -> None:
        """Worker that processes jobs from the queue."""
        while self._running:
            try:
                # Wait for a job with timeout to allow clean shutdown
                try:
                    job = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                if job.status == JobStatus.CANCELLED:
                    continue

                await self._process_job(job)

            except Exception as e:
                logger.exception("Worker error: %s", e)

    async def _process_job(self, job: Job) -> None:
        """Process a single job."""
        logger.info("Processing job %s", job.job_id)

        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)
        self._save_job(job)

        try:
            # Import here to avoid circular dependencies
            from diaremot.pipeline.audio_pipeline_core import run_pipeline

            # Create progress callback
            async def progress_callback(stage: str, progress: float, message: str):
                # Map stage name to enum
                stage_map = {
                    "dependency_check": (JobStage.DEPENDENCY_CHECK, 0),
                    "preprocess": (JobStage.PREPROCESS, 1),
                    "background_sed": (JobStage.BACKGROUND_SED, 2),
                    "diarize": (JobStage.DIARIZE, 3),
                    "transcribe": (JobStage.TRANSCRIBE, 4),
                    "paralinguistics": (JobStage.PARALINGUISTICS, 5),
                    "affect_and_assemble": (JobStage.AFFECT_AND_ASSEMBLE, 6),
                    "overlap_interruptions": (JobStage.OVERLAP_INTERRUPTIONS, 7),
                    "conversation_analysis": (JobStage.CONVERSATION_ANALYSIS, 8),
                    "speaker_rollups": (JobStage.SPEAKER_ROLLUPS, 9),
                    "outputs": (JobStage.OUTPUTS, 10),
                }

                if stage in stage_map:
                    stage_enum, stage_idx = stage_map[stage]
                    await job.update_progress(
                        stage=stage_enum,
                        stage_index=stage_idx,
                        stage_progress=progress,
                        message=message,
                    )

            # Note: run_pipeline is synchronous, need to run in executor
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                run_pipeline,
                str(job.input_path),
                str(job.output_dir),
                job.config,
            )

            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            job.results = results
            await job.update_progress(message="Job completed successfully")

            logger.info("Job %s completed successfully", job.job_id)

        except Exception as e:
            logger.exception("Job %s failed: %s", job.job_id, e)
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now(timezone.utc)
            job.error = str(e)
            await job.update_progress(message=f"Job failed: {e}")

        finally:
            self._save_job(job)


# Global job queue instance
_job_queue: JobQueue | None = None


def get_job_queue() -> JobQueue:
    """Get the global job queue instance."""
    global _job_queue
    if _job_queue is None:
        jobs_dir = Path("jobs")
        _job_queue = JobQueue(jobs_dir)
    return _job_queue


async def initialize_job_queue() -> None:
    """Initialize and start the job queue."""
    queue = get_job_queue()
    await queue.start()


async def shutdown_job_queue() -> None:
    """Shutdown the job queue."""
    global _job_queue
    if _job_queue:
        await _job_queue.stop()
