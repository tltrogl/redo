"""Tests for overlap summary stage wiring."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("numpy")


def _ensure_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module


def _load_module(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Cannot load module {name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
_ensure_package("diaremot", SRC_ROOT / "diaremot")
_ensure_package("diaremot.pipeline", SRC_ROOT / "diaremot/pipeline")
_ensure_package("diaremot.pipeline.stages", SRC_ROOT / "diaremot/pipeline/stages")

base_module = _load_module(
    "diaremot.pipeline.stages.base", SRC_ROOT / "diaremot/pipeline/stages/base.py"
)
summaries_module = _load_module(
    "diaremot.pipeline.stages.summaries",
    SRC_ROOT / "diaremot/pipeline/stages/summaries.py",
)

PipelineState = base_module.PipelineState
run_overlap = summaries_module.run_overlap


class _CoreLogStub:
    def __init__(self) -> None:
        self.stage_calls: list[tuple[str, str, dict[str, object]]] = []

    def stage(self, stage: str, level: str, **payload: object) -> None:
        self.stage_calls.append((stage, level, payload))


class _GuardStub:
    def __init__(self) -> None:
        self.done_calls: list[dict[str, int]] = []

    def done(self, **counts: int) -> None:
        self.done_calls.append(counts)


class _ParalinguisticsStub:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[list[dict[str, object]]] = []

    def compute_overlap_and_interruptions(self, turns):  # noqa: ANN001
        self.calls.append(turns)
        return self.response


def test_run_overlap_populates_state(tmp_path) -> None:  # noqa: ANN001
    payload = {
        "overlap_total_sec": 2.5,
        "overlap_ratio": 0.4,
        "by_speaker": {
            "A": {"overlap_sec": 1.5, "made": 2},
            "B": {"overlap_sec": 1.0, "made": 0},
        },
        "interruptions": [
            {"interrupter": "A", "interrupted": "B"},
            {"interrupter": "B", "interrupted": "A"},
        ],
    }

    paraling_stub = _ParalinguisticsStub(payload)
    pipeline = SimpleNamespace(
        paralinguistics_module=paraling_stub,
        corelog=_CoreLogStub(),
    )
    state = PipelineState(input_audio_path="input.wav", out_dir=tmp_path)
    state.turns = [
        {"speaker": "A", "start": 0.0, "end": 1.0},
        {"speaker": "B", "start": 0.6, "end": 1.4},
    ]
    guard = _GuardStub()

    run_overlap(pipeline, state, guard)

    assert paraling_stub.calls == [state.turns]
    assert state.overlap_stats == {"overlap_total_sec": 2.5, "overlap_ratio": 0.4}
    assert state.per_speaker_interrupts == {
        "A": {"made": 2, "received": 1, "overlap_sec": 1.5},
        "B": {"made": 0, "received": 1, "overlap_sec": 1.0},
    }
    assert guard.done_calls[-1] == {}
