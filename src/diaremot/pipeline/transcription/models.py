from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .backends import backends

__all__ = [
    "TranscriptionSegment",
    "BatchingConfig",
    "TranscriptionError",
    "resample_audio_fast",
    "estimate_snr_db",
]


@dataclass
class TranscriptionSegment:
    start_time: float
    end_time: float
    text: str
    confidence: float
    speaker_id: str | None = None
    speaker_name: str | None = None
    words: list[dict[str, Any]] | None = None
    language: str | None = None
    language_probability: float | None = None
    processing_time: float | None = None
    model_used: str | None = None
    asr_logprob_avg: float | None = None
    snr_db: float | None = None

    def __post_init__(self) -> None:
        self.start_time = max(0.0, float(self.start_time))
        self.end_time = max(self.start_time, float(self.end_time))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.text = str(self.text).strip()
        if self.speaker_id and not self.speaker_name:
            self.speaker_name = self.speaker_id

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "confidence": self.confidence,
            "speaker_id": self.speaker_id,
            "speaker_name": self.speaker_name,
            "words": self.words,
            "language": self.language,
            "language_probability": self.language_probability,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "asr_logprob_avg": self.asr_logprob_avg,
            "snr_db": self.snr_db,
            "duration": self.duration,
        }


@dataclass
class BatchingConfig:
    enabled: bool = True
    min_segments_threshold: int = 12
    short_segment_max_sec: float = 8.0
    batch_silence_sec: float = 0.3
    min_segment_sec: float = 0.05
    max_batch_duration_sec: float = 300
    target_batch_size_sec: float = 60.0
    max_segments_per_batch: int = 50


class TranscriptionError(Exception):
    """Raised when transcription cannot be completed."""


def resample_audio_fast(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    ratio = target_sr / orig_sr
    if ratio == 0.5 and orig_sr == 32000:
        return audio[::2].astype(np.float32)
    if ratio == 2.0 and orig_sr == 8000:
        return np.repeat(audio, 2).astype(np.float32)

    if backends.has_librosa:
        return backends.librosa.resample(
            audio.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr
        )

    new_length = int(len(audio) * ratio)
    old_indices = np.linspace(0, len(audio) - 1, new_length)
    return np.interp(old_indices, np.arange(len(audio)), audio).astype(np.float32)


def estimate_snr_db(audio: np.ndarray) -> float:
    if audio.size == 0:
        return float("nan")

    try:
        audio = audio.astype(np.float32)
        signal_power = float(np.mean(audio**2))
        noise_estimate = float(np.var(np.diff(audio)))
        snr = 10.0 * np.log10((signal_power + 1e-9) / (noise_estimate + 1e-9))
        return snr
    except Exception:
        return 10.0
