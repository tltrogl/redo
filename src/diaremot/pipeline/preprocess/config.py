"""Configuration and structured results for the preprocessing stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["PreprocessConfig", "AudioHealth", "PreprocessResult"]


@dataclass(slots=True)
class PreprocessConfig:
    """Runtime configuration for :class:`AudioPreprocessor`."""

    target_sr: int = 16000
    mono: bool = True

    # Auto-chunking for long audio files
    auto_chunk_enabled: bool = True
    chunk_threshold_minutes: float = 60.0  # Split audio longer than this
    chunk_size_minutes: float = 20.0  # Each chunk duration
    chunk_overlap_seconds: float = 30.0  # Overlap between chunks
    chunk_temp_dir: str | None = None  # Use system temp if None

    # High-pass
    hpf_hz: float = 80.0
    hpf_order: int = 2

    # Denoise (soft spectral subtraction with temporal smoothing + backoff)
    denoise: str = "spectral_sub_soft"  # "spectral_sub_soft" | "none"
    denoise_alpha_db: float = 3.0  # over-subtraction in dB
    denoise_beta: float = 0.06  # spectral floor as fraction of noise (0..1)
    mask_exponent: float = 1.0  # 1 ~ Wiener-ish
    smooth_t: int = 3  # median smoothing width (frames) for mask
    high_clip_backoff: float = 0.12  # backoff if floor_clipping_ratio exceeds this

    # Noise tracking
    noise_update_alpha: float = 0.10  # EMA for noise profile updates (lower = smoother)
    min_noise_frames: int = 30  # min non-speech frames to trust VAD noise estimate

    # VAD (RMS-gated; CPU-friendly)
    use_vad: bool = True
    frame_ms: int = 20
    hop_ms: int = 10
    vad_rel_db: float = 12.0  # speech if rms_db > noise_floor_db + vad_rel_db
    vad_floor_percentile: float = 20.0

    # Gated upward gain
    gate_db: float = -45.0  # below this, do not boost
    target_db: float = -23.0  # aim per-frame towards this
    max_boost_db: float = 18.0  # cap upward gain
    gain_smooth_ms: int = 250
    gain_smooth_method: str = "hann"  # "hann" | "exp"
    exp_smooth_alpha: float = 0.15

    # Compression (transparent)
    comp_ratio: float = 2.0
    comp_thresh_db: float = -26.0
    comp_knee_db: float = 6.0

    # Loudness norm (approximate)
    loudness_mode: str = "asr"  # "asr" -> hotter (-20 LUFS equiv), "broadcast" -> -23
    lufs_target_asr: float = -20.0
    lufs_target_broadcast: float = -23.0

    # QC / metrics
    oversample_factor: int = 4  # for intersample peak check
    silence_db: float = -60.0  # below counts as silence


@dataclass(slots=True)
class AudioHealth:
    snr_db: float
    clipping_detected: bool
    silence_ratio: float
    rms_db: float
    est_lufs: float
    dynamic_range_db: float
    floor_clipping_ratio: float
    is_chunked: bool = False
    chunk_info: dict[str, Any] | None = None


@dataclass(slots=True)
class PreprocessResult:
    """Structured result emitted by the preprocessing stack."""

    audio: np.ndarray
    sample_rate: int
    health: AudioHealth | None
    duration_s: float
    is_chunked: bool = False
    chunk_details: dict[str, Any] | None = None

    def to_tuple(self) -> tuple[np.ndarray, int, AudioHealth | None]:
        """Return the legacy tuple representation (audio, sr, health)."""

        return self.audio, self.sample_rate, self.health

    def __iter__(self):
        """Allow unpacking ``PreprocessResult`` like the historical tuple."""

        yield from (self.audio, self.sample_rate, self.health)
