"""Filtering, VAD, and spectral denoising primitives."""

from __future__ import annotations

import logging

import librosa
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)

__all__ = [
    "butter_highpass",
    "frame_params",
    "db",
    "rms_db",
    "percentile_db",
    "simple_vad",
    "spectral_subtract_soft_vad",
]


def butter_highpass(y: np.ndarray, sr: int, freq: float, order: int = 2) -> np.ndarray:
    if freq <= 0:
        return y
    nyq = 0.5 * sr
    Wn = min(0.999, max(1e-6, freq / nyq))
    b, a = butter(order, Wn, btype="high", analog=False)
    return filtfilt(b, a, y)


def db(x: float) -> float:
    return float(20.0 * np.log10(max(1e-12, x)))


def rms_db(y: np.ndarray) -> float:
    return db(float(np.sqrt(np.mean(np.square(y)) + 1e-12)))


def frame_params(sr: int, frame_ms: int, hop_ms: int) -> tuple[int, int]:
    n_fft = int(round(frame_ms * 0.001 * sr))
    n_fft = max(256, 1 << int(np.ceil(np.log2(max(8, n_fft)))))
    hop = int(round(hop_ms * 0.001 * sr))
    hop = max(1, min(hop, n_fft // 2))
    return n_fft, hop


def percentile_db(y_abs: np.ndarray, p: float) -> float:
    return db(float(np.percentile(y_abs, p)))


def simple_vad(
    y: np.ndarray, sr: int, frame_ms: int, hop_ms: int, floor_pct: float, rel_db: float
) -> np.ndarray:
    """Framewise RMS VAD; speech if above noise floor + margin."""

    n_fft, hop = frame_params(sr, frame_ms, hop_ms)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann")
    mag = np.abs(S)
    rms = np.sqrt(np.mean(mag**2, axis=0) + 1e-12)
    rms_db = 20 * np.log10(rms + 1e-12)
    floor = np.percentile(rms_db, floor_pct)
    speech = rms_db > (floor + rel_db)
    return speech.astype(np.bool_)


def spectral_subtract_soft_vad(
    y: np.ndarray,
    sr: int,
    speech_mask: np.ndarray | None,
    *,
    alpha_db: float,
    beta: float,
    mask_exponent: float,
    smooth_t: int,
    noise_ema_alpha: float,
    min_noise_frames: int,
    frame_ms: int,
    hop_ms: int,
    backoff_thresh: float,
) -> tuple[np.ndarray, float]:
    """Soft spectral subtraction with optional VAD guidance."""

    n_fft, hop = frame_params(sr, frame_ms, hop_ms)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann")
    mag, phase = np.abs(S), np.angle(S)

    if speech_mask is not None and np.sum(~speech_mask) >= max(min_noise_frames, 5):
        noise_mag = np.median(mag[:, ~speech_mask], axis=1, keepdims=True)
    else:
        noise_mag = np.percentile(mag, 10, axis=1, keepdims=True)

    alpha = 10.0 ** (alpha_db / 20.0)

    residual = mag - alpha * noise_mag
    floor = beta * noise_mag
    clean_mag = np.maximum(residual, floor)

    M = (clean_mag / (clean_mag + alpha * noise_mag + 1e-12)) ** mask_exponent

    if smooth_t > 1:
        M = median_filter(M, size=(1, smooth_t))

    floor_hits = (residual <= floor).sum()
    total_bins = residual.size
    floor_ratio = float(floor_hits) / float(total_bins + 1e-12)

    if floor_ratio > backoff_thresh:
        alpha *= 0.75
        beta2 = min(0.08, beta * 1.25)
        residual2 = mag - alpha * noise_mag
        floor2 = beta2 * noise_mag
        clean_mag2 = np.maximum(residual2, floor2)
        M = (clean_mag2 / (clean_mag2 + alpha * noise_mag + 1e-12)) ** mask_exponent
        if smooth_t > 1:
            M = median_filter(M, size=(1, smooth_t))
        floor_hits = (residual2 <= floor2).sum()
        total_bins = residual2.size
        floor_ratio = float(floor_hits) / float(total_bins + 1e-12)
        logger.warning("[denoise] High floor clipping; applied backoff (ratio=%.3f)", floor_ratio)

    S_hat = M * mag * np.exp(1j * phase)
    y_hat = librosa.istft(S_hat, hop_length=hop, window="hann", length=len(y))
    return y_hat.astype(np.float32), float(floor_ratio)
