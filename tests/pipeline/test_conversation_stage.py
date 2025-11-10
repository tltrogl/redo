"""Tests for the conversation analysis pipeline stage."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

_STAGE_STUBS = (
    "affect",
    "asr",
    "dependency_check",
    "diarize",
    "paralinguistics",
    "preprocess",
)


def _make_stage_stub(name: str) -> ModuleType:
    module = ModuleType(name)

    def _noop(*_args: object, **_kwargs: object) -> None:
        return None

    if name.endswith(".preprocess"):
        module.run_preprocess = _noop  # type: ignore[attr-defined]
        module.run_background_sed = _noop  # type: ignore[attr-defined]
    else:
        module.run = _noop  # type: ignore[attr-defined]
    return module


for _stub in _STAGE_STUBS:
    _full_name = f"diaremot.pipeline.stages.{_stub}"
    if _full_name not in sys.modules:
        sys.modules[_full_name] = _make_stage_stub(_full_name)

from diaremot.pipeline.stages import summaries
from diaremot.summaries.conversation_analysis import (
    ConversationAnalysisError,
    ConversationMetrics,
)


class _DummyCoreLog:
    def __init__(self) -> None:
        self.stage_calls: list[tuple[str, str, dict[str, object]]] = []
        self.event_calls: list[tuple[str, str, dict[str, object]]] = []

    def stage(self, stage: str, status: str, **payload: object) -> None:  # noqa: D401 - simple stub
        """Store stage invocations for later inspection."""

        self.stage_calls.append((stage, status, dict(payload)))

    def event(self, stage: str, event: str, **payload: object) -> None:  # noqa: D401 - simple stub
        """Store event invocations for later inspection."""

        self.event_calls.append((stage, event, dict(payload)))


class _DummyPipeline:
    def __init__(self) -> None:
        self.corelog = _DummyCoreLog()


class _DummyGuard:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def done(self, **counts: object) -> None:  # noqa: D401 - simple stub
        """Capture guard completion invocations."""

        self.calls.append(dict(counts))


def test_run_conversation_logs_warning_without_stdout(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Failures in the analysis stage should log via corelog without printing."""

    def _raise_error(*_args: object, **_kwargs: object) -> ConversationMetrics:
        raise ConversationAnalysisError("synthetic failure")

    monkeypatch.setattr(summaries, "analyze_conversation_flow", _raise_error)

    pipeline = _DummyPipeline()
    state = SimpleNamespace(segments_final=[{"start": 0.0, "end": 1.0}], duration_s=30.0, conv_metrics=None)
    guard = _DummyGuard()

    summaries.run_conversation(pipeline, state, guard)

    captured = capsys.readouterr()
    assert captured.out == ""

    assert pipeline.corelog.stage_calls, "corelog.stage should be invoked on failure"
    stage_name, status, payload = pipeline.corelog.stage_calls[0]
    assert stage_name == "conversation_analysis"
    assert status == "warn"
    assert "synthetic failure" in str(payload.get("message"))

    assert guard.calls, "guard.done should be called despite the failure"
    assert isinstance(state.conv_metrics, ConversationMetrics)
