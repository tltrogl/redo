"""Signal chain utilities used by :class:`AudioPreprocessor`."""

from __future__ import annotations

import logging

import librosa
import numpy as np

from .config import AudioHealth, PreprocessConfig, PreprocessResult
from .denoise import (
    butter_highpass,
    frame_params,
    percentile_db,
    rms_db,
    simple_vad,
    spectral_subtract_soft_vad,
)

logger = logging.getLogger(__name__)

__all__ = [
    "process_array",
    "combine_chunk_health",
]


def hann_smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    w = np.hanning(win)
    w = w / (w.sum() + 1e-12)
    pad = win // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xpad, w, mode="valid")


def exp_smooth(x: np.ndarray, alpha: float) -> np.ndarray:
    y = np.empty_like(x)
    acc = x[0]
    for i, v in enumerate(x):
        acc = alpha * v + (1 - alpha) * acc
        y[i] = acc
    return y


def interp_per_sample(env: np.ndarray, hop: int, length: int) -> np.ndarray:
    t_env = np.arange(len(env)) * hop
    t = np.arange(length)
    return np.interp(t, t_env, env, left=env[0], right=env[-1])


def oversampled_clip_detect(y: np.ndarray, factor: int = 4, thresh: float = 0.999) -> bool:
    if factor <= 1:
        return bool(np.any(np.abs(y) >= thresh))
    idx = np.arange(len(y), dtype=np.float64)
    fine = np.linspace(0, len(y) - 1, num=(len(y) - 1) * factor + 1)
    y2 = np.interp(fine, idx, y.astype(np.float64))
    return bool(np.any(np.abs(y2) >= thresh))


def dynamic_range_db(y: np.ndarray) -> float:
    y_abs = np.abs(y) + 1e-12
    hi = percentile_db(y_abs, 95.0)
    lo = percentile_db(y_abs, 5.0)
    return max(0.0, hi - lo)


def estimate_loudness_lufs_approx(y: np.ndarray, sr: int) -> float:
    win = int(0.400 * sr)
    hop = int(0.100 * sr)
    if win <= 0 or len(y) < win:
        return rms_db(y)
    frames = librosa.util.frame(y, frame_length=win, hop_length=hop).T
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    loud = 20 * np.log10(rms + 1e-12)
    ungated_mean = np.mean(loud)
    gated = loud[loud > (ungated_mean - 10.0)]
    lufs = float(np.mean(gated) if len(gated) else ungated_mean)
    return lufs


def apply_upward_gain(y: np.ndarray, n_fft: int, hop: int, config: PreprocessConfig) -> np.ndarray:
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann")
    mag = np.abs(S)
    frame_rms = np.sqrt(np.mean(mag**2, axis=0) + 1e-12)
    frame_db = 20 * np.log10(frame_rms + 1e-12)

    gain_db = np.zeros_like(frame_db)
    gate_db = float(config.gate_db)
    target_db = float(config.target_db)
    max_boost = float(config.max_boost_db)
    needs_boost = (frame_db > gate_db) & (frame_db < target_db)
    gain_db[needs_boost] = np.minimum(target_db - frame_db[needs_boost], max_boost)

    smooth_len = max(1, int(round(config.gain_smooth_ms / config.hop_ms)))
    if config.gain_smooth_method == "hann":
        gain_db_sm = hann_smooth(gain_db, smooth_len)
    else:
        gain_db_sm = exp_smooth(gain_db, alpha=float(config.exp_smooth_alpha))

    gain_lin = np.power(10.0, gain_db_sm / 20.0)
    env = interp_per_sample(gain_lin, hop, len(y))
    return y * env.astype(np.float32)


def apply_compression(y: np.ndarray, n_fft: int, hop: int, config: PreprocessConfig) -> np.ndarray:
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann")
    mag = np.abs(S)
    lvl_db = 20 * np.log10(np.sqrt(np.mean(mag**2, axis=0)) + 1e-12)

    thr = float(config.comp_thresh_db)
    ratio = float(config.comp_ratio)
    knee = float(config.comp_knee_db)

    over = lvl_db - thr
    comp_gain_db = np.zeros_like(over)
    lower = -knee / 2.0
    upper = knee / 2.0
    for i, o in enumerate(over):
        if o <= lower:
            comp_gain_db[i] = 0.0
        elif o < upper:
            t = (o - lower) / (knee + 1e-12)
            desired = thr + o / ratio
            comp_gain_db[i] = (desired - (thr + o)) * t
        else:
            comp_gain_db[i] = (thr + o / ratio) - (thr + o)

    comp_gain_lin = np.power(10.0, comp_gain_db / 20.0)
    comp_env = interp_per_sample(comp_gain_lin, hop, len(y))
    return y * comp_env.astype(np.float32)


def apply_loudness(y: np.ndarray, sr: int, config: PreprocessConfig) -> np.ndarray:
    current_lufs = estimate_loudness_lufs_approx(y, sr)
    target_lufs = (
        config.lufs_target_asr if config.loudness_mode == "asr" else config.lufs_target_broadcast
    )
    loudness_gain_db = np.clip(target_lufs - current_lufs, -12.0, 12.0)
    loudness_gain_lin = 10.0 ** (loudness_gain_db / 20.0)
    return y * float(loudness_gain_lin)


def apply_safety_limit(y: np.ndarray) -> np.ndarray:
    if y.size == 0:
        return y.astype(np.float32)
    peak = float(np.max(np.abs(y)))
    if peak > 0.95:
        safety_gain = 0.95 / peak
        y = y * safety_gain
        logger.warning("Applied safety limiting: %.1f dB", 20 * np.log10(safety_gain))
    return y.astype(np.float32)


def build_health(y: np.ndarray, sr: int, config: PreprocessConfig, floor_ratio: float) -> AudioHealth:
    signal_power = float(np.mean(y**2)) if y.size else 0.0
    noise_estimate = float(np.percentile(y**2, 10)) if y.size else 0.0
    snr_db = 10 * np.log10(signal_power / max(noise_estimate, 1e-12)) if signal_power > 0 else 0.0

    silence_thresh = 10.0 ** (config.silence_db / 20.0)
    silence_frames = float(np.sum(np.abs(y) < silence_thresh)) if y.size else 0.0
    silence_ratio = silence_frames / float(len(y)) if len(y) > 0 else 1.0

    clipping_detected = oversampled_clip_detect(y, config.oversample_factor)
    dynamic_range = dynamic_range_db(y)
    rms = rms_db(y)
    est_lufs = estimate_loudness_lufs_approx(y, sr)

    return AudioHealth(
        snr_db=float(snr_db),
        clipping_detected=bool(clipping_detected),
        silence_ratio=float(silence_ratio),
        rms_db=float(rms),
        est_lufs=float(est_lufs),
        dynamic_range_db=float(dynamic_range),
        floor_clipping_ratio=float(floor_ratio),
    )


def combine_chunk_health(chunk_healths: list[AudioHealth], num_chunks: int) -> AudioHealth | None:
    if not chunk_healths:
        if num_chunks <= 0:
            return None
        return AudioHealth(
            snr_db=0.0,
            clipping_detected=False,
            silence_ratio=1.0,
            rms_db=-60.0,
            est_lufs=-60.0,
            dynamic_range_db=0.0,
            floor_clipping_ratio=0.0,
            is_chunked=True,
        )

    avg_snr = float(np.mean([h.snr_db for h in chunk_healths]))
    any_clipping = any(h.clipping_detected for h in chunk_healths)
    avg_silence = float(np.mean([h.silence_ratio for h in chunk_healths]))
    avg_rms = float(np.mean([h.rms_db for h in chunk_healths]))
    avg_lufs = float(np.mean([h.est_lufs for h in chunk_healths]))
    avg_dynamic_range = float(np.mean([h.dynamic_range_db for h in chunk_healths]))
    max_floor_clipping = float(np.max([h.floor_clipping_ratio for h in chunk_healths]))

    return AudioHealth(
        snr_db=avg_snr,
        clipping_detected=any_clipping,
        silence_ratio=avg_silence,
        rms_db=avg_rms,
        est_lufs=avg_lufs,
        dynamic_range_db=avg_dynamic_range,
        floor_clipping_ratio=max_floor_clipping,
        is_chunked=True,
    )


def process_array(y: np.ndarray, sr: int, config: PreprocessConfig) -> PreprocessResult:
    if y is None or len(y) == 0:
        empty = np.zeros(1, dtype=np.float32)
        return PreprocessResult(
            audio=empty,
            sample_rate=sr,
            health=None,
            duration_s=0.0,
            is_chunked=False,
        )

    y = np.asarray(y, dtype=np.float32)

    y_hp = butter_highpass(y, sr, config.hpf_hz, config.hpf_order)
    speech_mask = (
        simple_vad(
            y_hp,
            sr,
            config.frame_ms,
            config.hop_ms,
            floor_pct=config.vad_floor_percentile,
            rel_db=config.vad_rel_db,
        )
        if config.use_vad
        else None
    )

    if config.denoise == "spectral_sub_soft":
        y_denoised, floor_ratio = spectral_subtract_soft_vad(
            y_hp,
            sr,
            speech_mask,
            alpha_db=config.denoise_alpha_db,
            beta=config.denoise_beta,
            mask_exponent=config.mask_exponent,
            smooth_t=config.smooth_t,
            noise_ema_alpha=config.noise_update_alpha,
            min_noise_frames=config.min_noise_frames,
            frame_ms=config.frame_ms,
            hop_ms=config.hop_ms,
            backoff_thresh=config.high_clip_backoff,
        )
    else:
        y_denoised = y_hp
        floor_ratio = 0.0

    n_fft, hop = frame_params(sr, config.frame_ms, config.hop_ms)
    y_boosted = apply_upward_gain(y_denoised, n_fft, hop, config)
    y_compressed = apply_compression(y_boosted, n_fft, hop, config)
    y_loud = apply_loudness(y_compressed, sr, config)
    y_final = apply_safety_limit(y_loud)

    health = build_health(y_final, sr, config, floor_ratio)
    duration_s = len(y_final) / sr if sr else 0.0

    return PreprocessResult(
        audio=y_final,
        sample_rate=sr,
        health=health,
        duration_s=float(duration_s),
        is_chunked=False,
    )
