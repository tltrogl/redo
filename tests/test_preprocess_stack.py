from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
librosa = pytest.importorskip("librosa")

from diaremot.pipeline.preprocess.chain import (
    SpectralFrameStats,
    apply_compression,
    apply_upward_gain,
    build_health,
    exp_smooth,
    hann_smooth,
    interp_per_sample,
)
from diaremot.pipeline.preprocess.chunking import ChunkInfo, merge_chunked_audio
from diaremot.pipeline.preprocess.config import PreprocessConfig
from diaremot.pipeline.preprocess.denoise import frame_params, spectral_subtract_soft_vad


def test_merge_chunked_audio_trims_overlap() -> None:
    sr = 16000
    chunk_a = np.ones(sr, dtype=np.float32)
    chunk_b = np.full(sr, 2.0, dtype=np.float32)

    info_a = ChunkInfo(
        chunk_id=0,
        start_time=0.0,
        end_time=1.0,
        duration=1.0,
        overlap_start=0.0,
        overlap_end=0.25,
        temp_path="/tmp/chunk0.wav",
    )
    info_b = ChunkInfo(
        chunk_id=1,
        start_time=1.0,
        end_time=2.0,
        duration=1.0,
        overlap_start=0.25,
        overlap_end=0.0,
        temp_path="/tmp/chunk1.wav",
    )

    merged = merge_chunked_audio([(chunk_a, info_a), (chunk_b, info_b)], sr)
    expected = sr + sr - int(info_b.overlap_start * sr)
    assert len(merged) == expected
    assert np.allclose(merged[: sr], chunk_a)
    assert np.allclose(merged[sr:], chunk_b[int(info_b.overlap_start * sr) :])


def test_spectral_subtract_soft_vad_reduces_energy() -> None:
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    signal = 0.2 * np.sin(2 * np.pi * 440 * t)
    rng = np.random.default_rng(0)
    noisy = signal + 0.05 * rng.standard_normal(sr)

    cleaned, floor_ratio = spectral_subtract_soft_vad(
        noisy.astype(np.float32),
        sr,
        speech_mask=None,
        alpha_db=3.0,
        beta=0.06,
        mask_exponent=1.0,
        smooth_t=3,
        noise_ema_alpha=0.1,
        min_noise_frames=30,
        frame_ms=20,
        hop_ms=10,
        backoff_thresh=0.12,
    )

    assert cleaned.shape == noisy.shape
    assert 0.0 <= floor_ratio <= 1.0
    assert np.mean(cleaned**2) <= np.mean(noisy**2) * 1.05


def test_build_health_on_silence() -> None:
    config = PreprocessConfig()
    sr = config.target_sr
    silence = np.zeros(sr, dtype=np.float32)

    health = build_health(silence, sr, config, floor_ratio=0.0)

    assert health.silence_ratio == 1.0
    assert not health.clipping_detected
    assert health.dynamic_range_db == 0.0
    assert health.snr_db == 0.0
    assert health.floor_clipping_ratio == 0.0


def test_upward_gain_and_compression_match_legacy_fft() -> None:
    config = PreprocessConfig()
    sr = config.target_sr
    t = np.linspace(0, 2.0, int(2.0 * sr), endpoint=False)
    signal = 0.3 * np.sin(2 * np.pi * 220 * t)
    mod = 0.5 * np.sin(2 * np.pi * 2 * t)
    noisy = signal * (1.0 + 0.5 * mod) + 0.02 * np.random.default_rng(42).standard_normal(t.shape)
    y = noisy.astype(np.float32)

    n_fft, hop = frame_params(sr, config.frame_ms, config.hop_ms)

    def legacy_apply_upward_gain(y_in: np.ndarray) -> np.ndarray:
        S = librosa.stft(y_in, n_fft=n_fft, hop_length=hop, window="hann")
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
        env = interp_per_sample(gain_lin, hop, len(y_in))
        return y_in * env.astype(np.float32)

    def legacy_apply_compression(y_in: np.ndarray) -> np.ndarray:
        S = librosa.stft(y_in, n_fft=n_fft, hop_length=hop, window="hann")
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
                t_val = (o - lower) / (knee + 1e-12)
                desired = thr + o / ratio
                comp_gain_db[i] = (desired - (thr + o)) * t_val
            else:
                comp_gain_db[i] = (thr + o / ratio) - (thr + o)

        comp_gain_lin = np.power(10.0, comp_gain_db / 20.0)
        comp_env = interp_per_sample(comp_gain_lin, hop, len(y_in))
        return y_in * comp_env.astype(np.float32)

    legacy_boosted = legacy_apply_upward_gain(y)
    legacy_compressed = legacy_apply_compression(legacy_boosted)

    boosted_no_cache, boosted_stats_no_cache = apply_upward_gain(y, n_fft, hop, config)
    compressed_no_cache = apply_compression(boosted_no_cache, n_fft, hop, config)

    spectral = SpectralFrameStats.from_signal(y, n_fft, hop)
    boosted, boosted_stats = apply_upward_gain(y, n_fft, hop, config, spectral=spectral)
    compressed = apply_compression(boosted, n_fft, hop, config, spectral=boosted_stats)

    direct_boosted_stats_no_cache = SpectralFrameStats.from_signal(boosted_no_cache, n_fft, hop)
    assert np.allclose(
        boosted_stats_no_cache.frame_db, direct_boosted_stats_no_cache.frame_db, atol=1e-6
    )

    assert np.allclose(boosted_no_cache, legacy_boosted, atol=1e-5)
    assert np.allclose(compressed_no_cache, legacy_compressed, atol=1e-5)

    direct_boosted_stats = SpectralFrameStats.from_signal(boosted, n_fft, hop)
    assert np.allclose(boosted_stats.frame_db, direct_boosted_stats.frame_db, atol=1e-6)

    assert np.allclose(boosted, legacy_boosted, atol=1e-5)
    assert np.allclose(compressed, legacy_compressed, atol=1e-5)
