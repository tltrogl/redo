from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def _ensure_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module


ROOT = Path(__file__).resolve().parents[2] / "src"
_ensure_package("diaremot", ROOT / "diaremot")
_ensure_package("diaremot.pipeline", ROOT / "diaremot/pipeline")
_ensure_package("diaremot.pipeline.core", ROOT / "diaremot/pipeline/core")
_ensure_package("diaremot.pipeline.stages", ROOT / "diaremot/pipeline/stages")


def _load_module(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Cannot load module {name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


affect_mixin_module = _load_module(
    "diaremot.pipeline.core.affect_mixin", ROOT / "diaremot/pipeline/core/affect_mixin.py"
)
stage_module = _load_module(
    "diaremot.pipeline.stages.affect", ROOT / "diaremot/pipeline/stages/affect.py"
)

AffectMixin = affect_mixin_module.AffectMixin
_SegmentAudioFactory = stage_module._SegmentAudioFactory


class _DummyAffect:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray | None, int, str]] = []

    def analyze(self, wav: np.ndarray | None, sr: int, text: str) -> dict[str, int | str | None]:
        self.calls.append((wav, sr, text))
        return {"length": 0 if wav is None else int(wav.size), "text": text}


class _DummyStats:
    def __init__(self) -> None:
        self.issues: list[str] = []


class _DummyLogger:
    def warn(self, _message: str) -> None:  # pragma: no cover - defensive
        raise AssertionError("Unexpected warning during test")


class _DummyPipeline(AffectMixin):
    def __init__(self) -> None:
        self.affect = _DummyAffect()
        self.stats = _DummyStats()
        self.corelog = _DummyLogger()


def test_segment_audio_window_shares_buffer() -> None:
    source = np.linspace(-1.0, 1.0, num=128, dtype=np.float32)
    factory = _SegmentAudioFactory(source)
    window = factory.segment(10, 40)

    view = window.as_array(copy=False)
    assert np.shares_memory(view, source)

    mem = window.as_memoryview()
    assert mem is not None

    # Mutations through the NumPy view update the original buffer as well.
    view[0] = 42.0
    assert float(source[10]) == 42.0


def test_affect_unified_accepts_segment_window() -> None:
    pipeline = _DummyPipeline()
    data = np.arange(0, 32, dtype=np.float32)
    window = _SegmentAudioFactory(data).segment(4, 12)

    result = pipeline._affect_unified(window, 16000, "hello")
    assert result["length"] == 8
    wav, sr, text = pipeline.affect.calls[-1]
    assert sr == 16000
    assert text == "hello"
    assert isinstance(wav, np.ndarray)
    assert wav.shape[0] == 8
    assert np.shares_memory(wav, data)


def test_affect_unified_accepts_iterables() -> None:
    pipeline = _DummyPipeline()

    result = pipeline._affect_unified((0.1 for _ in range(5)), 16000, "iterable")
    assert result["length"] == 5
    wav, _, _ = pipeline.affect.calls[-1]
    assert isinstance(wav, np.ndarray)
    assert wav.dtype == np.float32
