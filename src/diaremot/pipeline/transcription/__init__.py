"""Transcription subpackage exposing backend management and scheduling primitives."""

from .backends import (
    BackendAvailability,
    ModelManager,
    backends,
    configure_environment,
    get_system_capabilities,
)
from .models import (
    BatchingConfig,
    TranscriptionError,
    TranscriptionSegment,
    estimate_snr_db,
    estimate_snr_db_cached,
    resample_audio_fast,
)
from .postprocess import distribute_batch_results, distribute_text_proportionally
from .scheduler import (
    AsyncTranscriber,
    AudioTranscriber,
    benchmark_transcription,
    create_batch_groups,
    create_transcriber,
)

__all__ = [
    "BackendAvailability",
    "ModelManager",
    "backends",
    "configure_environment",
    "get_system_capabilities",
    "BatchingConfig",
    "TranscriptionError",
    "TranscriptionSegment",
    "estimate_snr_db",
    "estimate_snr_db_cached",
    "resample_audio_fast",
    "distribute_batch_results",
    "distribute_text_proportionally",
    "AsyncTranscriber",
    "AudioTranscriber",
    "create_transcriber",
    "benchmark_transcription",
    "create_batch_groups",
]
