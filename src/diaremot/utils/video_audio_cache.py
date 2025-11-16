"""Helpers for extracting and caching audio tracks from video containers."""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {
    ".mp4",
    ".m4v",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".mpg",
    ".mpeg",
    ".wmv",
    ".flv",
}

__all__ = ["VIDEO_EXTENSIONS", "is_probably_video", "ensure_cached_audio"]


def is_probably_video(path: Path | str) -> bool:
    """Return True when ``path`` looks like a video container we should demux."""

    suffix = Path(path).suffix.lower()
    return suffix in VIDEO_EXTENSIONS


def ensure_cached_audio(
    source: Path | str,
    *,
    cache_dir: Path | str,
    target_sr: int,
    channels: int = 1,
    ffmpeg_bin: str | None = None,
) -> Path:
    """Extract ``source`` audio track once and return the cached WAV path."""

    src_path = Path(source).expanduser().resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"Video source {src_path} is missing")

    cache_root = Path(cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    stat = src_path.stat()
    key = f"{src_path}::{stat.st_size}::{stat.st_mtime_ns}::{target_sr}::{channels}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    cached = cache_root / f"{src_path.stem}.{digest[:16]}.wav"
    if cached.exists():
        return cached

    tmp_path = cached.with_suffix(cached.suffix + ".tmp")
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except OSError:
            pass

    ffmpeg = ffmpeg_bin or os.getenv("FFMPEG_BIN") or "ffmpeg"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src_path),
        "-vn",
        "-ac",
        str(max(1, channels)),
        "-ar",
        str(target_sr),
        "-f",
        "wav",
        "-loglevel",
        "error",
        str(tmp_path),
    ]

    try:
        logger.info("Extracting audio track from %s → %s", src_path.name, cached.name)
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        tmp_path.replace(cached)
        return cached
    except subprocess.CalledProcessError as exc:  # pragma: no cover - relies on ffmpeg
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg failed to extract audio: {stderr.strip()}") from exc
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
