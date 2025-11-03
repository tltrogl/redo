"""Façade for the transcription subsystem."""

from __future__ import annotations

import asyncio
import functools
import inspect
from collections.abc import Sequence
from typing import Any

import numpy as np

from .transcription import (
    AsyncTranscriber,
    BatchingConfig,
    TranscriptionSegment,
    configure_environment,
    get_system_capabilities,
)
from .transcription import (
    AudioTranscriber as _AudioTranscriber,
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
        self._async_enabled = bool(enable_async)
        if self._async_enabled:
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
        audio = np.asarray(audio_16k_mono, dtype=np.float32)
        diar_list = list(diar_segments)
        result = self._engine.transcribe_segments(audio, sr, diar_list)
        if inspect.isawaitable(result):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    raise RuntimeError("event loop already running")
            except RuntimeError:
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    return loop.run_until_complete(result)
                finally:
                    asyncio.set_event_loop(None)
                    loop.close()
            else:
                return loop.run_until_complete(result)
        return result

    async def transcribe_segments_async(
        self,
        audio_16k_mono: np.ndarray,
        sr: int,
        diar_segments: Sequence[dict[str, Any]],
    ) -> list[TranscriptionSegment]:
        audio = np.asarray(audio_16k_mono, dtype=np.float32)
        diar_list = list(diar_segments)
        engine = self._engine
        candidate = getattr(engine, "transcribe_segments_async", None)
        if callable(candidate):
            result = candidate(audio, sr, diar_list)
            if inspect.isawaitable(result):
                return await result
            return result

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(engine.transcribe_segments, audio, sr, diar_list),
        )

    @property
    def async_enabled(self) -> bool:
        return self._async_enabled and isinstance(self._engine, AsyncTranscriber)

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
