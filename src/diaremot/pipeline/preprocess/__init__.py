"""Preprocessing primitives used by :mod:`diaremot.pipeline` stages."""

from .chain import combine_chunk_health, process_array
from .chunking import (
    ChunkedMemmapAssembler,
    ChunkInfo,
    cleanup_chunks,
    create_audio_chunks,
    merge_chunked_audio,
)
from .config import AudioHealth, PreprocessConfig, PreprocessResult
from .io import decode_audio_segment, probe_audio_metadata, safe_load_audio

__all__ = [
    "AudioHealth",
    "PreprocessConfig",
    "PreprocessResult",
    "ChunkInfo",
    "probe_audio_metadata",
    "safe_load_audio",
    "decode_audio_segment",
    "create_audio_chunks",
    "merge_chunked_audio",
    "cleanup_chunks",
    "process_array",
    "combine_chunk_health",
    "ChunkedMemmapAssembler",
]
