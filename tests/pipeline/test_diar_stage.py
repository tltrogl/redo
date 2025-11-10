from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from diaremot.pipeline.stages.base import PipelineState
from diaremot.pipeline.stages.diarize import run as run_diarize


PP_SIG = json.dumps({"key": "value"}, sort_keys=True)


class _StubGuard:
    def __init__(self) -> None:
        self.progress_calls: list[str] = []
        self.done_calls: list[dict[str, int]] = []

    def progress(self, message: str) -> None:
        self.progress_calls.append(message)

    def done(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.done_calls.append(kwargs)


class _StubCoreLog:
    def stage(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        return None

    def event(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        return None


class _StubCheckpoints:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    def create_checkpoint(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.calls.append((args, kwargs))


class _StubDiarizer:
    def __init__(self, turns: list[dict[str, float]], stats: dict[str, float]) -> None:
        self._turns = turns
        self._stats = stats
        self.calls = 0

    def diarize_audio(self, *_args, **_kwargs) -> list[dict[str, float]]:
        self.calls += 1
        return self._turns

    def get_vad_statistics(self) -> dict[str, float]:
        return dict(self._stats)


class _StubPipeline:
    def __init__(self, diarizer: _StubDiarizer) -> None:
        self.diar = diarizer
        self.corelog = _StubCoreLog()
        self.checkpoints = _StubCheckpoints()
        self.cache_version = "test"
        self.cfg: dict[str, object] = {}


def _build_state(tmp_path: Path) -> PipelineState:
    state = PipelineState(input_audio_path="input.wav", out_dir=tmp_path)
    state.y = np.zeros(16000, dtype=np.float32)
    state.sr = 16000
    state.duration_s = 60.0
    state.audio_sha16 = "abc123"
    state.pp_sig = PP_SIG
    state.cache_dir = tmp_path
    return state


def test_diar_stage_uses_backend_vad_metric_and_caches(tmp_path: Path) -> None:
    turns = [{"start": 0.0, "end": 1.0, "speaker": "Speaker_1"}]
    stats = {"vad_boundary_flips": 120.0, "speech_regions": 5.0}
    pipeline = _StubPipeline(_StubDiarizer(turns, stats))
    state = _build_state(tmp_path)
    guard = _StubGuard()

    run_diarize(pipeline, state, guard)

    assert state.turns == turns
    assert state.vad_unstable is True
    cache_payload = json.loads((tmp_path / "diar.json").read_text(encoding="utf-8"))
    assert cache_payload["diagnostics"]["vad_boundary_flips"] == 120.0


def test_diar_stage_reads_cached_vad_metric(tmp_path: Path) -> None:
    turns = [{"start": 0.0, "end": 2.0, "speaker": "Speaker_2"}]
    cache = {
        "version": "test",
        "audio_sha16": "abc123",
        "pp_signature": PP_SIG,
        "turns": turns,
        "diagnostics": {"vad_boundary_flips": 10.0},
    }
    pipeline = _StubPipeline(_StubDiarizer([], {"vad_boundary_flips": 0.0}))
    state = _build_state(tmp_path)
    state.duration_s = 600.0
    state.resume_diar = True
    state.diar_cache = cache
    guard = _StubGuard()

    run_diarize(pipeline, state, guard)

    assert pipeline.diar.calls == 0
    assert state.turns == turns
    assert state.vad_unstable is False
