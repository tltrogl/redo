import numpy as np

from diaremot.pipeline.diarization.config import DiarizationConfig
from diaremot.pipeline.diarization.pipeline import SpeakerDiarizer


def test_diarize_audio_accepts_speech_regions():
    cfg = DiarizationConfig()
    diar = SpeakerDiarizer(cfg)
    # Very short silence-only waveform
    wav = np.zeros(16000, dtype=np.float32)
    # Provide a dummy speech_region — pipeline should handle gracefully
    regions = [(0.0, 0.5)]
    turns = diar.diarize_audio(wav, 16000, speech_regions=regions)
    assert isinstance(turns, list)
    stats = diar.get_vad_statistics()
    assert stats.get("speech_regions") == float(len(regions))


def test_diarize_audio_populates_debug_payload():
    cfg = DiarizationConfig()
    diar = SpeakerDiarizer(cfg)
    wav = np.zeros(16000, dtype=np.float32)
    regions = [(0.0, 0.6)]
    diar.diarize_audio(
        wav,
        16000,
        speech_regions=regions,
        region_source="unit_test",
    )
    payload = diar.get_debug_payload()
    assert payload["vad"]["region_count"] == len(regions)
    assert payload["input"]["speech_region_source"] == "unit_test"
    assert payload["turns"]["speaker_count"] >= 1
