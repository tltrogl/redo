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
