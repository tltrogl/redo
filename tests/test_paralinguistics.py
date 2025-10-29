import json

import numpy as np
import pytest

from diaremot.affect.paralinguistics import (
    PARSELMOUTH_AVAILABLE,
    ParalinguisticsConfig,
    compute_segment_features_v2,
)


def test_parselmouth_used_even_when_snr_low():
    if not PARSELMOUTH_AVAILABLE:
        pytest.skip("Parselmouth not installed in test environment")

    sr = 16000
    duration_sec = 1.0
    samples = int(sr * duration_sec)
    time_axis = np.linspace(0.0, duration_sec, samples, endpoint=False)

    # Quiet fundamental masked by comparatively loud noise to trigger low SNR heuristics.
    speech = 0.01 * np.sin(2 * np.pi * 130.0 * time_axis)
    noise = 0.2 * np.random.default_rng(0).standard_normal(samples)
    audio = (speech + noise).astype(np.float32)

    cfg = ParalinguisticsConfig(vq_min_duration_sec=0.5)
    features = compute_segment_features_v2(audio, sr, 0.0, duration_sec, "", cfg)

    flags = json.loads(features["paralinguistics_flags_json"])
    voice_quality_flags = flags.get("voice_quality", {})

    assert voice_quality_flags.get("method") == "parselmouth"
    assert not features["vq_reliable"]
    assert features["vq_note"].startswith("low_quality_")
