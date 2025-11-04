"""Chunk management helpers for long-form recordings."""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import get_audio_duration, probe_audio_metadata

logger = logging.getLogger(__name__)

__all__ = [
    "ChunkInfo",
    "create_audio_chunks",
    "merge_chunked_audio",
    "cleanup_chunks",
    "ChunkedMemmapAssembler",
]


@dataclass(slots=True)
class ChunkInfo:
    chunk_id: int
    start_time: float
    end_time: float
    duration: float
    overlap_start: float
    overlap_end: float
    temp_path: str | None = None


def create_audio_chunks(
    audio_path: str,
    config,
    *,
    duration: float | None = None,
    info=None,
) -> list[ChunkInfo]:
    """Plan overlapping chunks for long recordings without decoding upfront."""

    logger.info("[chunks] Planning audio chunks for long file: %s", audio_path)

    if duration is None or duration <= 0:
        duration = get_audio_duration(audio_path, info=info)

    if info is None:
        _, info = probe_audio_metadata(audio_path)

    logger.info(
        "[chunks] Audio duration: %.1f minutes; threshold=%s min; size=%s min; overlap=%ss",
        duration / 60.0,
        config.chunk_threshold_minutes,
        config.chunk_size_minutes,
        config.chunk_overlap_seconds,
    )

    chunk_duration = config.chunk_size_minutes * 60.0
    overlap_duration = config.chunk_overlap_seconds

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

        chunk_info = ChunkInfo(
            chunk_id=chunk_id,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            overlap_start=start_time - actual_start,
            overlap_end=actual_end - end_time,
            temp_path=None,
        )
        chunks.append(chunk_info)

        start_time = end_time
        chunk_id += 1

    logger.info("[chunks] Planned %s chunks", len(chunks))
    return chunks


class ChunkedMemmapAssembler:
    """Utility to assemble processed chunks into a contiguous memmap."""

    def __init__(self, chunks: list[ChunkInfo], target_sr: int):
        self.target_sr = int(target_sr)
        max_end = max((c.end_time + c.overlap_end) for c in chunks) if chunks else 0.0
        # Allocate slightly more to account for rounding during resampling.
        total_samples = int(np.ceil(max_end * self.target_sr)) + self.target_sr
        total_samples = max(total_samples, 1)
        tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        self.path = Path(tmp.name)
        tmp.close()
        self.mem = np.lib.format.open_memmap(
            self.path,
            mode="w+",
            dtype=np.float32,
            shape=(total_samples,),
        )
        self.mem[:] = 0.0
        self.max_written = 0

    def write(self, audio: np.ndarray, chunk: ChunkInfo) -> None:
        audio = np.asarray(audio, dtype=np.float32)
        expected_start = int(round(chunk.start_time * self.target_sr))
        expected_end = int(round(chunk.end_time * self.target_sr))
        expected_len = max(expected_end - expected_start, 0)

        trim_start = int(round(chunk.overlap_start * self.target_sr))
        trim_end = int(round(chunk.overlap_end * self.target_sr))

        if trim_start > 0:
            audio = audio[trim_start:]
        if trim_end > 0:
            audio = audio[:-trim_end] if trim_end <= audio.shape[0] else np.zeros(0, dtype=np.float32)

        if expected_len and audio.shape[0] != expected_len:
            if audio.shape[0] > expected_len:
                audio = audio[:expected_len]
            else:
                audio = np.pad(audio, (0, expected_len - audio.shape[0]), mode="constant")

        if expected_start + audio.shape[0] > self.mem.shape[0]:
            raise RuntimeError("Chunk assembler overflow; increase allocation margin")

        end_idx = expected_start + audio.shape[0]
        if audio.size:
            self.mem[expected_start:end_idx] = audio
        self.max_written = max(self.max_written, end_idx)

    def finalize(self) -> tuple[np.memmap, Path, int]:
        original_len = self.mem.shape[0]
        self.mem.flush()
        total = max(int(self.max_written), 0)
        del self.mem
        src = np.lib.format.open_memmap(self.path, mode="r", dtype=np.float32)
        if 0 <= total < original_len:
            tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
            tmp.close()
            if total > 0:
                trimmed = np.lib.format.open_memmap(
                    tmp.name,
                    mode="w+",
                    dtype=np.float32,
                    shape=(total,),
                )
                trimmed[:] = src[:total]
                trimmed.flush()
                del trimmed
            else:
                np.save(tmp.name, np.zeros(0, dtype=np.float32))
            del src
            os.replace(tmp.name, self.path)
            mem = np.lib.format.open_memmap(self.path, mode="r", dtype=np.float32)
        else:
            mem = src
        view = mem[:total] if total else mem[:0]
        return view, self.path, total


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
        if not chunk.temp_path:
            continue
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
