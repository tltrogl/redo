"""Preprocessing primitives used by :mod:`diaremot.pipeline` stages."""

from .config import AudioHealth, PreprocessConfig, PreprocessResult
from .io import probe_audio_metadata, safe_load_audio
from .chunking import ChunkInfo, cleanup_chunks, create_audio_chunks, merge_chunked_audio
from .chain import combine_chunk_health, process_array

__all__ = [
    "AudioHealth",
    "PreprocessConfig",
    "PreprocessResult",
    "ChunkInfo",
    "probe_audio_metadata",
    "safe_load_audio",
    "create_audio_chunks",
    "merge_chunked_audio",
    "cleanup_chunks",
    "process_array",
    "combine_chunk_health",
]
