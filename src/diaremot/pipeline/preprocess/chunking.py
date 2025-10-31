"""Chunk management helpers for long-form recordings."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .io import get_audio_duration, is_uncompressed_pcm, probe_audio_metadata

logger = logging.getLogger(__name__)

__all__ = [
    "ChunkInfo",
    "create_audio_chunks",
    "merge_chunked_audio",
    "cleanup_chunks",
]


@dataclass(slots=True)
class ChunkInfo:
    chunk_id: int
    start_time: float
    end_time: float
    duration: float
    overlap_start: float
    overlap_end: float
    temp_path: str


def create_audio_chunks(
    audio_path: str,
    config,
    *,
    duration: float | None = None,
    info: sf.Info | None = None,
) -> list[ChunkInfo]:
    """Create overlapping chunks on disk for long recordings."""

    logger.info("[chunks] Creating audio chunks for long file: %s", audio_path)

    if duration is None or duration <= 0:
        duration = get_audio_duration(audio_path, info=info)

    if info is None:
        _, info = probe_audio_metadata(audio_path)

    if info and info.samplerate:
        sr = int(info.samplerate)
    else:
        sr = int(getattr(config, "target_sr", 16000))

    logger.info(
        "[chunks] Audio duration: %.1f minutes; threshold=%s min; size=%s min; overlap=%ss",
        duration / 60.0,
        config.chunk_threshold_minutes,
        config.chunk_size_minutes,
        config.chunk_overlap_seconds,
    )

    chunk_duration = config.chunk_size_minutes * 60.0
    overlap_duration = config.chunk_overlap_seconds

    if config.chunk_temp_dir:
        temp_dir = Path(config.chunk_temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="audio_chunks_"))

    chunks: list[ChunkInfo] = []
    chunk_id = 0
    start_time = 0.0

    while start_time < duration:
        end_time = min(start_time + chunk_duration, duration)

        if chunk_id > 0:
            actual_start = max(0.0, start_time - overlap_duration)
        else:
            actual_start = start_time

        if end_time < duration:
            actual_end = min(duration, end_time + overlap_duration)
        else:
            actual_end = end_time

        temp_chunk_raw = None
        try:
            t0 = time.time()
            temp_chunk_raw = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_chunk_raw.close()

            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                audio_path,
                "-ss",
                str(actual_start),
                "-t",
                str(actual_end - actual_start),
                "-ac",
                "1",
                "-ar",
                str(sr),
                "-loglevel",
                "quiet",
                temp_chunk_raw.name,
            ]

            result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                chunk_audio, _ = sf.read(temp_chunk_raw.name, dtype="float32")
                if chunk_audio.ndim > 1:
                    chunk_audio = np.mean(chunk_audio, axis=1)
                chunk_audio = chunk_audio.astype(np.float32)
                logger.info(
                    "[chunks] Extracted chunk %s via ffmpeg in %.2fs (%.1fs→%.1fs)",
                    chunk_id,
                    time.time() - t0,
                    actual_start,
                    actual_end,
                )
            else:
                raise subprocess.CalledProcessError(result.returncode, cmd)
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ):
            if info and is_uncompressed_pcm(info):
                logger.debug(
                    "ffmpeg chunk extraction failed; reading PCM chunk via soundfile for chunk %s",
                    chunk_id,
                )
                frames_start = int(round(actual_start * sr))
                frames_end = int(round(actual_end * sr))
                frames = max(frames_end - frames_start, 0)
                with sf.SoundFile(audio_path) as snd:
                    snd.seek(frames_start)
                    chunk_audio = snd.read(frames, dtype="float32", always_2d=False)
                if chunk_audio.ndim > 1:
                    chunk_audio = np.mean(chunk_audio, axis=1)
                chunk_audio = chunk_audio.astype(np.float32)
                logger.info(
                    "[chunks] Extracted chunk %s via soundfile in %.2fs (%.1fs→%.1fs)",
                    chunk_id,
                    time.time() - t0,
                    actual_start,
                    actual_end,
                )
            else:
                raise RuntimeError(
                    "ffmpeg chunk extraction failed and only PCM WAV/AIFF fallback is supported"
                )
        finally:
            if temp_chunk_raw is not None:
                try:
                    os.unlink(temp_chunk_raw.name)
                except Exception:
                    pass

        chunk_filename = f"chunk_{chunk_id:03d}_{int(start_time):04d}s-{int(end_time):04d}s.wav"
        chunk_path = temp_dir / chunk_filename
        sf.write(chunk_path, chunk_audio, sr)

        chunk_info = ChunkInfo(
            chunk_id=chunk_id,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            overlap_start=start_time - actual_start,
            overlap_end=actual_end - end_time,
            temp_path=str(chunk_path),
        )
        chunks.append(chunk_info)

        logger.info("[chunks] Saved chunk %s: %s (%.1fs)", chunk_id, chunk_filename, chunk_info.duration)

        start_time = end_time
        chunk_id += 1

    logger.info("[chunks] Created %s chunks in %s", len(chunks), temp_dir)
    return chunks


def merge_chunked_audio(chunks: list[tuple[np.ndarray, ChunkInfo]], target_sr: int) -> np.ndarray:
    """Merge processed chunks back into a single waveform."""

    logger.info("Merging %s processed chunks", len(chunks))

    if not chunks:
        return np.array([], dtype=np.float32)

    if len(chunks) == 1:
        return chunks[0][0]

    chunks.sort(key=lambda x: x[1].start_time)
    merged_parts: list[np.ndarray] = []

    for i, (chunk_audio, chunk_info) in enumerate(chunks):
        if i == 0:
            merged_parts.append(chunk_audio)
            continue
        overlap_samples = int(chunk_info.overlap_start * target_sr)
        if overlap_samples < len(chunk_audio):
            chunk_audio = chunk_audio[overlap_samples:]
        merged_parts.append(chunk_audio)

    merged = np.concatenate(merged_parts, axis=0)
    logger.info("Merged audio: %.1fs total", len(merged) / max(target_sr, 1))
    return merged.astype(np.float32)


def cleanup_chunks(chunks: Iterable[ChunkInfo]) -> None:
    """Robust cleanup of temporary chunk files with retry logic."""

    chunk_list = list(chunks)
    logger.info("Cleaning up %s temporary chunk files", len(chunk_list))
    temp_dirs: set[Path] = set()
    failed_cleanups: list[Path] = []

    for chunk in chunk_list:
        chunk_path = Path(chunk.temp_path)
        if chunk_path.exists():
            temp_dirs.add(chunk_path.parent)
            try:
                chunk_path.unlink()
            except (OSError, PermissionError) as exc:
                logger.warning("Failed to remove %s: %s", chunk_path, exc)
                failed_cleanups.append(chunk_path)

    if failed_cleanups:
        time.sleep(0.1)
        for chunk_path in failed_cleanups:
            try:
                if chunk_path.exists():
                    chunk_path.unlink()
            except Exception as exc:  # pragma: no cover - best effort cleanup
                logger.warning("Retried removal failed for %s: %s", chunk_path, exc)

    for temp_dir in temp_dirs:
        try:
            if temp_dir.exists() and not any(temp_dir.iterdir()):
                temp_dir.rmdir()
        except Exception as exc:  # pragma: no cover - best effort cleanup
            logger.warning("Could not remove temp directory %s: %s", temp_dir, exc)
