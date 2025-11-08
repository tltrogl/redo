"""Legacy façade around the refactored preprocessing package."""

from __future__ import annotations

import logging

import numpy as np
import soundfile as sf

from .preprocess import (
    AudioHealth,
    ChunkInfo,
    PreprocessConfig,
    PreprocessResult,
    ChunkedMemmapAssembler,
    combine_chunk_health,
    create_audio_chunks,
    probe_audio_metadata,
    safe_load_audio,
)
from .preprocess.io import decode_audio_segment
from .preprocess import (
    process_array as run_preprocess_array,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AudioPreprocessor",
    "AudioHealth",
    "PreprocessConfig",
    "PreprocessResult",
]


class AudioPreprocessor:
    """Coordinator that drives the modular preprocessing pipeline."""

    def __init__(self, config: PreprocessConfig | None = None):
        self.config = config or PreprocessConfig()

    def process_file(self, path: str) -> PreprocessResult:
        """Load ``path`` and run the preprocessing chain."""

        duration, info = probe_audio_metadata(path)
        threshold_seconds = self.config.chunk_threshold_minutes * 60.0

        if self.config.auto_chunk_enabled and duration >= threshold_seconds:
            logger.info(
                "Long audio detected (%.1fmin), auto-chunking into ~%d min windows",
                duration / 60.0,
                int(self.config.chunk_size_minutes),
            )
            return self._process_file_chunked(path, duration, info)

        logger.info("Processing audio normally (%.1fmin)", duration / 60.0)
        y, sr = safe_load_audio(path, target_sr=self.config.target_sr, mono=self.config.mono)
        return self.process_array(y, sr)

    def _process_file_chunked(
        self,
        path: str,
        duration: float,
        info: sf.Info | None,
    ) -> PreprocessResult:
        chunks_info = create_audio_chunks(
            path,
            self.config,
            duration=duration,
            info=info,
        )

        if not chunks_info:
            logger.warning("No chunks created, falling back to normal processing")
            y, sr = safe_load_audio(path, target_sr=self.config.target_sr, mono=self.config.mono)
            return self.process_array(y, sr)

        assembler = ChunkedMemmapAssembler(chunks_info, self.config.target_sr)
        chunk_healths: list[AudioHealth] = []

        for chunk_info in chunks_info:
            logger.info("Processing chunk %s/%s", chunk_info.chunk_id, len(chunks_info) - 1)
            actual_start = max(0.0, chunk_info.start_time - chunk_info.overlap_start)
            actual_end = chunk_info.end_time + chunk_info.overlap_end
            segment_duration = max(0.0, actual_end - actual_start)
            if segment_duration == 0.0:
                continue

            y_chunk = decode_audio_segment(
                path,
                self.config.target_sr,
                mono=self.config.mono,
                start=actual_start,
                duration=segment_duration,
                info=info,
            )
            chunk_result = self.process_array(y_chunk, self.config.target_sr)
            assembler.write(chunk_result.audio, chunk_info)
            if chunk_result.health:
                chunk_healths.append(chunk_result.health)

        _, storage_path, num_samples = assembler.finalize()

        chunk_meta = {
            "num_chunks": len(chunks_info),
            "chunk_duration_minutes": self.config.chunk_size_minutes,
            "total_duration_minutes": num_samples / self.config.target_sr / 60.0,
            "overlap_seconds": self.config.chunk_overlap_seconds,
            "storage_path": str(storage_path),
        }

        combined_health = combine_chunk_health(chunk_healths, len(chunks_info))
        if combined_health:
            combined_health.is_chunked = True
            combined_health.chunk_info = chunk_meta

        duration_s = num_samples / self.config.target_sr if self.config.target_sr else 0.0

        logger.info("Chunked processing complete: %.1fs total", duration_s)

        return PreprocessResult(
            audio=None,
            sample_rate=self.config.target_sr,
            health=combined_health,
            duration_s=float(duration_s),
            is_chunked=True,
            chunk_details=chunk_meta,
            audio_path=str(storage_path),
            num_samples=int(num_samples),
        )

    def process_array(self, y: np.ndarray, sr: int) -> PreprocessResult:
        return run_preprocess_array(y, sr, self.config)
