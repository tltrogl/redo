from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from diaremot.pipeline.core import paralinguistics_mixin as mixin


class _PipelineStub(mixin.ParalinguisticsMixin):
    def __init__(self) -> None:
        self.corelog = SimpleNamespace(warn=lambda *args, **kwargs: None)


def test_extract_paraling_maps_extended_voice_quality(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = SimpleNamespace(
        extract=lambda wav, sr, segs: [
            {
                "wpm": 180,
                "duration_s": 0.5,
                "words": 2,
                "pause_count": 1,
                "pause_time_s": 0.2,
                "pause_ratio": 0.4,
                "f0_mean_hz": 210.0,
                "f0_std_hz": 15.0,
                "loudness_rms": 0.12,
                "disfluency_count": 1,
                "vq_jitter_pct": 0.7,
                "vq_shimmer_db": 0.9,
                "vq_hnr_db": 12.3,
                "vq_cpps_db": 14.5,
                "vq_voiced_ratio": "0.75",
                "vq_spectral_slope_db": "-8.5",
                "vq_reliable": "false",
                "vq_note": "breathy",
            }
        ]
    )
    monkeypatch.setattr(mixin, "para", stub)

    pipeline = _PipelineStub()
    wav = np.ones(800, dtype=np.float32)
    segs = [{"start": 0.0, "end": 0.5, "text": "hello world"}]

    metrics = pipeline._extract_paraling(wav, 16000, segs)

    assert metrics[0]["vq_voiced_ratio"] == pytest.approx(0.75)
    assert metrics[0]["vq_spectral_slope_db"] == pytest.approx(-8.5)
    assert metrics[0]["vq_reliable"] is False
    assert metrics[0]["vq_note"] == "breathy"


def test_extract_paraling_fallback_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mixin, "para", None)

    pipeline = _PipelineStub()
    wav = np.linspace(-1.0, 1.0, 1600, dtype=np.float32)
    segs = [{"start": 0.0, "end": 0.1, "text": "fallback"}]

    metrics = pipeline._extract_paraling(wav, 16000, segs)

    assert metrics[0]["vq_voiced_ratio"] == 0.0
    assert metrics[0]["vq_spectral_slope_db"] == 0.0
    assert metrics[0]["vq_reliable"] is False
    assert metrics[0]["vq_note"] == ""
