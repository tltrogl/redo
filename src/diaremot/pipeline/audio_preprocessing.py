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
    cleanup_chunks,
    combine_chunk_health,
    create_audio_chunks,
    merge_chunked_audio,
    probe_audio_metadata,
    safe_load_audio,
)
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

        processed_chunks: list[tuple[np.ndarray, ChunkInfo]] = []
        chunk_healths: list[AudioHealth] = []

        try:
            for chunk_info in chunks_info:
                logger.info("Processing chunk %s/%s", chunk_info.chunk_id, len(chunks_info) - 1)
                y_chunk, sr = safe_load_audio(
                    chunk_info.temp_path,
                    target_sr=self.config.target_sr,
                    mono=self.config.mono,
                )
                chunk_result = self.process_array(y_chunk, sr)
                processed_chunks.append((chunk_result.audio, chunk_info))
                if chunk_result.health:
                    chunk_healths.append(chunk_result.health)

            merged_audio = merge_chunked_audio(processed_chunks, self.config.target_sr)

            chunk_meta = {
                "num_chunks": len(chunks_info),
                "chunk_duration_minutes": self.config.chunk_size_minutes,
                "total_duration_minutes": len(merged_audio) / self.config.target_sr / 60.0,
                "overlap_seconds": self.config.chunk_overlap_seconds,
            }

            combined_health = combine_chunk_health(chunk_healths, len(chunks_info))
            if combined_health:
                combined_health.is_chunked = True
                combined_health.chunk_info = chunk_meta

            duration_s = len(merged_audio) / self.config.target_sr if self.config.target_sr else 0.0

            logger.info(
                "Chunked processing complete: %.1fs total",
                len(merged_audio) / self.config.target_sr,
            )

            return PreprocessResult(
                audio=merged_audio.astype(np.float32),
                sample_rate=self.config.target_sr,
                health=combined_health,
                duration_s=float(duration_s),
                is_chunked=True,
                chunk_details=chunk_meta,
            )
        finally:
            cleanup_chunks(chunks_info)

    def process_array(self, y: np.ndarray, sr: int) -> PreprocessResult:
        return run_preprocess_array(y, sr, self.config)
