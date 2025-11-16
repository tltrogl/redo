"""Unit tests for the paralinguistics pipeline stage."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"


def _ensure_package(name: str, path: Path) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = module
    return module


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, SRC_ROOT / relative_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"Unable to load module {name} from {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ensure_package("diaremot", SRC_ROOT / "diaremot")
_ensure_package("diaremot.pipeline", SRC_ROOT / "diaremot/pipeline")
_ensure_package("diaremot.pipeline.stages", SRC_ROOT / "diaremot/pipeline/stages")

_load_module("diaremot.pipeline.logging_utils", "diaremot/pipeline/logging_utils.py")
base_module = _load_module(
    "diaremot.pipeline.stages.base", "diaremot/pipeline/stages/base.py"
)
paralinguistics = _load_module(
    "diaremot.pipeline.stages.paralinguistics",
    "diaremot/pipeline/stages/paralinguistics.py",
)

PipelineState = base_module.PipelineState


class _GuardStub:
    def __init__(self) -> None:
        self.progress_calls: list[dict[str, object]] = []
        self.done_calls: list[dict[str, int]] = []

    def progress(
        self,
        message: str,
        *,
        step: int | None = None,
        total: int | None = None,
    ) -> None:
        self.progress_calls.append(
            {"message": message, "step": step, "total": total}
        )

    def done(self, **kwargs: int) -> None:
        self.done_calls.append(kwargs)


class _PipelineStub:
    def __init__(self) -> None:
        self.stats = SimpleNamespace(config_snapshot={})
        self.extract_calls: list[tuple[np.ndarray, int, list[dict[str, object]]]] = []
        self.return_value: dict[int, dict[str, object]] | list[dict[str, object]] = {}

    def _extract_paraling(
        self, wav, sr: int, segs: list[dict[str, object]]
    ):  # noqa: ANN001
        self.extract_calls.append((wav, sr, segs))
        return self.return_value


def _make_state(tmp_path) -> PipelineState:  # noqa: ANN001
    state = PipelineState(input_audio_path="input.wav", out_dir=tmp_path)
    state.y = np.ones(1600, dtype=np.float32)
    state.sr = 16000
    return state


def test_stage_skips_when_no_transcript(tmp_path) -> None:  # noqa: ANN001
    pipeline = _PipelineStub()
    state = _make_state(tmp_path)
    guard = _GuardStub()

    paralinguistics.run(pipeline, state, guard)

    assert pipeline.extract_calls == []
    assert state.para_metrics == {}
    assert guard.progress_calls[-1]["message"].startswith("skip: no transcript")
    assert guard.done_calls[-1]["count"] == 0


def test_stage_skips_when_upstream_failed(tmp_path) -> None:  # noqa: ANN001
    pipeline = _PipelineStub()
    pipeline.stats.config_snapshot["transcribe_failed"] = True
    state = _make_state(tmp_path)
    state.norm_tx = [{"start": 0.0, "end": 0.5, "text": "hi"}]
    guard = _GuardStub()

    paralinguistics.run(pipeline, state, guard)

    assert pipeline.extract_calls == []
    assert guard.progress_calls[-1]["message"].startswith("skip: upstream")
    assert guard.done_calls[-1]["count"] == 0


def test_stage_passes_audio_view_for_many_segments(tmp_path) -> None:  # noqa: ANN001
    pipeline = _PipelineStub()
    state = _make_state(tmp_path)
    state.norm_tx = [
        {"start": i * 0.5, "end": i * 0.5 + 0.25, "text": f"seg-{i}"}
        for i in range(16)
    ]
    expected_metrics = {i: {"wpm": float(i)} for i in range(len(state.norm_tx))}
    pipeline.return_value = expected_metrics
    guard = _GuardStub()

    paralinguistics.run(pipeline, state, guard)

    assert len(pipeline.extract_calls) == 1
    wav_arg, sr_arg, segs_arg = pipeline.extract_calls[0]
    assert wav_arg is state.y
    assert sr_arg == state.sr
    assert segs_arg is state.norm_tx
    assert state.para_metrics == expected_metrics
    assert guard.done_calls[-1]["count"] == len(expected_metrics)
