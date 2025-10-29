"""Façade for the transcription subsystem."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .transcription import (
    AsyncTranscriber,
    AudioTranscriber as _AudioTranscriber,
    BatchingConfig,
    TranscriptionSegment,
    configure_environment,
    get_system_capabilities,
)

__all__ = [
    "Transcriber",
    "TranscriptionSegment",
    "BatchingConfig",
    "get_system_capabilities",
    "create_transcriber",
    "AudioTranscriber",
    "AsyncTranscriber",
]


class Transcriber:
    """Pipeline-oriented façade around the transcription engine."""

    def __init__(
        self,
        *,
        enable_async: bool = False,
        enable_batching: bool = True,
        batching_config: BatchingConfig | None = None,
        max_workers: int = 2,
        **kwargs: Any,
    ) -> None:
        configure_environment()
        self._batching = batching_config or BatchingConfig(enabled=enable_batching)
        base_kwargs = {"batching_config": self._batching, "max_workers": max_workers}
        base_kwargs.update(kwargs)
        if enable_async:
            self._engine: AsyncTranscriber | _AudioTranscriber = AsyncTranscriber(
                **base_kwargs
            )
        else:
            self._engine = _AudioTranscriber(**base_kwargs)

    @property
    def engine(self) -> AsyncTranscriber | _AudioTranscriber:
        return self._engine

    def transcribe_segments(
        self,
        audio_16k_mono: np.ndarray,
        sr: int,
        diar_segments: Sequence[dict[str, Any]],
    ) -> list[TranscriptionSegment]:
        return self._engine.transcribe_segments(
            np.asarray(audio_16k_mono, dtype=np.float32),
            sr,
            list(diar_segments),
        )

    def validate_backend(self) -> dict[str, Any]:
        return self._engine.validate_backend()

    def get_model_info(self) -> dict[str, Any]:
        return self._engine.get_model_info()

    def get_performance_stats(self) -> dict[str, Any]:
        if hasattr(self._engine, "get_performance_stats"):
            return self._engine.get_performance_stats()
        return {}


def create_transcriber(*args: Any, **kwargs: Any) -> Transcriber:
    """Backwards-compatible factory returning the façade."""
    return Transcriber(*args, **kwargs)


AudioTranscriber = _AudioTranscriber
