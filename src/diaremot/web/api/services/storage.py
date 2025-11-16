"""File storage service for uploads and results."""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import shutil
import uuid
from pathlib import Path
from typing import Any

import soundfile as sf

logger = logging.getLogger(__name__)


class StorageService:
    """Manages file uploads and storage."""

    def __init__(self, upload_dir: Path, output_dir: Path):
        self.upload_dir = upload_dir
        self.output_dir = output_dir

        # Create directories
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, filename: str, content: bytes) -> tuple[str, Path, dict[str, Any]]:
        """Save uploaded file.

        Args:
            filename: Original filename.
            content: File content.

        Returns:
            Tuple of (file_id, file_path, metadata).
        """
        # Generate unique file ID
        file_id = str(uuid.uuid4())

        # Get file extension
        suffix = Path(filename).suffix or ""

        # Save file
        file_path = self.upload_dir / f"{file_id}{suffix}"
        file_path.write_bytes(content)

        # Extract metadata
        metadata = {
            "file_id": file_id,
            "filename": filename,
            "size": len(content),
            "content_type": mimetypes.guess_type(filename)[0] or "application/octet-stream",
        }

        # Try to extract audio metadata
        try:
            audio_metadata = self._get_audio_metadata(file_path)
            metadata.update(audio_metadata)
        except Exception as e:
            logger.warning("Failed to extract audio metadata from %s: %s", filename, e)

        logger.info("Saved upload %s (%s) as %s", filename, metadata["size"], file_id)
        return file_id, file_path, metadata

    def _get_audio_metadata(self, file_path: Path) -> dict[str, Any]:
        """Extract audio metadata using soundfile.

        Args:
            file_path: Path to audio file.

        Returns:
            Dictionary with duration, sample_rate, channels.
        """
        try:
            info = sf.info(str(file_path))
            return {
                "duration": float(info.duration),
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
            }
        except Exception:
            # Try with ffmpeg metadata if soundfile fails
            return self._get_audio_metadata_ffmpeg(file_path)

    def _get_audio_metadata_ffmpeg(self, file_path: Path) -> dict[str, Any]:
        """Extract audio metadata using ffmpeg/ffprobe.

        Args:
            file_path: Path to audio file.

        Returns:
            Dictionary with duration, sample_rate, channels.
        """
        import json
        import subprocess

        try:
            # Try ffprobe first
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_format",
                    "-show_streams",
                    str(file_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)

                # Find audio stream
                audio_stream = next(
                    (s for s in data.get("streams", []) if s.get("codec_type") == "audio"),
                    None,
                )

                if audio_stream:
                    duration = float(data.get("format", {}).get("duration", 0))
                    sample_rate = int(audio_stream.get("sample_rate", 0))
                    channels = int(audio_stream.get("channels", 0))

                    return {
                        "duration": duration,
                        "sample_rate": sample_rate,
                        "channels": channels,
                    }

        except Exception as e:
            logger.warning("ffprobe failed for %s: %s", file_path, e)

        return {}

    def get_upload_path(self, file_id: str) -> Path | None:
        """Get path to uploaded file.

        Args:
            file_id: File identifier.

        Returns:
            Path to file or None if not found.
        """
        # Search for file with this ID
        for file_path in self.upload_dir.glob(f"{file_id}*"):
            return file_path
        return None

    def get_output_dir(self, job_id: str) -> Path:
        """Get output directory for a job.

        Args:
            job_id: Job identifier.

        Returns:
            Path to output directory.
        """
        output_path = self.output_dir / job_id
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    def get_result_file(self, job_id: str, filename: str) -> Path | None:
        """Get path to a result file.

        Args:
            job_id: Job identifier.
            filename: Result filename.

        Returns:
            Path to result file or None if not found.
        """
        output_dir = self.output_dir / job_id
        result_path = output_dir / filename
        if result_path.exists():
            return result_path
        return None

    def list_result_files(self, job_id: str) -> list[dict[str, Any]]:
        """List all result files for a job.

        Args:
            job_id: Job identifier.

        Returns:
            List of file info dictionaries.
        """
        output_dir = self.output_dir / job_id
        if not output_dir.exists():
            return []

        files = []
        for file_path in output_dir.iterdir():
            if file_path.is_file():
                files.append(
                    {
                        "filename": file_path.name,
                        "size": file_path.stat().st_size,
                        "content_type": mimetypes.guess_type(file_path.name)[0]
                        or "application/octet-stream",
                    }
                )

        return files

    def cleanup_upload(self, file_id: str) -> bool:
        """Delete uploaded file.

        Args:
            file_id: File identifier.

        Returns:
            True if deleted, False if not found.
        """
        file_path = self.get_upload_path(file_id)
        if file_path and file_path.exists():
            file_path.unlink()
            logger.info("Deleted upload %s", file_id)
            return True
        return False

    def cleanup_job_outputs(self, job_id: str) -> bool:
        """Delete all outputs for a job.

        Args:
            job_id: Job identifier.

        Returns:
            True if deleted, False if not found.
        """
        output_dir = self.output_dir / job_id
        if output_dir.exists():
            shutil.rmtree(output_dir)
            logger.info("Deleted outputs for job %s", job_id)
            return True
        return False


# Global storage instance
_storage: StorageService | None = None


def get_storage() -> StorageService:
    """Get the global storage instance."""
    global _storage
    if _storage is None:
        upload_dir = Path("uploads")
        output_dir = Path("outputs")
        _storage = StorageService(upload_dir, output_dir)
    return _storage
