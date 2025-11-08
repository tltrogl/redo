"""Regression tests for streaming-aware preprocessing."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from diaremot.pipeline.audio_preprocessing import AudioPreprocessor
from diaremot.pipeline.preprocess.config import PreprocessConfig
from diaremot.pipeline.stages.base import PipelineState
from diaremot.pipeline.stages.preprocess import run_preprocess


def _sine_wave(duration_s: float, sr: int) -> np.ndarray:
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
    return (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_chunked_preprocess_streams_to_memmap(tmp_path) -> None:
    sr = 16_000
    wav = _sine_wave(6.0, sr)
    audio_path = tmp_path / "long.wav"
    sf.write(audio_path, wav, sr)

    cfg = PreprocessConfig(
        auto_chunk_enabled=True,
        chunk_threshold_minutes=0.01,
        chunk_size_minutes=0.005,
        chunk_overlap_seconds=0.25,
        target_sr=sr,
    )

    pre = AudioPreprocessor(cfg)
    result = pre.process_file(str(audio_path))

    assert result.is_chunked
    assert result.audio is None
    assert result.audio_path is not None
    storage = np.load(result.audio_path, mmap_mode="r")
    assert storage.shape[0] == result.num_samples
    assert result.chunk_details and result.chunk_details.get("storage_path") == result.audio_path


class _DummyGuard:
    def progress(self, *_args, **_kwargs) -> None:  # pragma: no cover - simple stub
        pass

    def done(self, **_kwargs) -> None:  # pragma: no cover - simple stub
        pass


class _DummyCheckpoints:
    def seed_file_hash(self, *_args, **_kwargs) -> None:  # pragma: no cover
        pass

    def create_checkpoint(self, *_args, **_kwargs) -> None:  # pragma: no cover
        pass


class _DummyCorelog:
    def event(self, *_args, **_kwargs) -> None:  # pragma: no cover
        pass

    def stage(self, *_args, **_kwargs) -> None:  # pragma: no cover
        pass


class _DummyStats:
    file_id: str | None = None
    config_snapshot: dict[str, object]

    def __init__(self) -> None:
        self.config_snapshot = {}


def test_run_preprocess_defers_audio_until_needed(tmp_path) -> None:
    sr = 16_000
    wav = _sine_wave(5.0, sr)
    audio_path = tmp_path / "deferred.wav"
    sf.write(audio_path, wav, sr)

    cfg = PreprocessConfig(
        auto_chunk_enabled=True,
        chunk_threshold_minutes=0.01,
        chunk_size_minutes=0.004,
        chunk_overlap_seconds=0.2,
        target_sr=sr,
    )

    class _Pipeline:
        def __init__(self, root: str) -> None:
            self.pre = AudioPreprocessor(cfg)
            self.pp_conf = self.pre.config
            self.cache_root = tmp_path / "cache"
            self.cache_root.mkdir()
            self.cache_roots = [self.cache_root]
            self.cache_version = "test"
            self.checkpoints = _DummyCheckpoints()
            self.corelog = _DummyCorelog()
            self.cfg: dict[str, object] = {}
            self.stats = _DummyStats()

    pipeline = _Pipeline(str(tmp_path))
    state = PipelineState(str(audio_path), tmp_path)
    guard = _DummyGuard()

    run_preprocess(pipeline, state, guard)

    assert state.y.size == 0
    assert state.preprocessed_audio_path is not None
    assert state.preprocessed_audio_path.exists()
    assert state.preprocessed_num_samples and state.preprocessed_num_samples > 0

    # Ensure lazy loading works when a later stage needs the waveform.
    audio = state.ensure_audio()
    assert isinstance(audio, np.ndarray)
    assert audio.shape[0] == state.preprocessed_num_samples
