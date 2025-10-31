from __future__ import annotations

import numpy as np

from diaremot.pipeline.preprocess.chain import build_health
from diaremot.pipeline.preprocess.chunking import ChunkInfo, merge_chunked_audio
from diaremot.pipeline.preprocess.config import PreprocessConfig
from diaremot.pipeline.preprocess.denoise import spectral_subtract_soft_vad


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
