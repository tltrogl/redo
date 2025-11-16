"""Paralinguistics extraction stage."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ..logging_utils import StageGuard
from .base import PipelineState

if TYPE_CHECKING:
    from ..orchestrator import AudioAnalysisPipelineV2

__all__ = ["run"]


def _coerce_positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _coerce_int(value: Any) -> int | None:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _normalise_metrics(
    segments: list[dict[str, Any]], raw_metrics: dict[int, dict[str, Any]] | None
) -> dict[int, dict[str, Any]]:
    """Ensure baseline paralinguistic metrics are always present."""

    normalised: dict[int, dict[str, Any]] = {}
    source = raw_metrics if isinstance(raw_metrics, dict) else {}

    for idx, segment in enumerate(segments):
        entry = source.get(idx)
        data = dict(entry) if isinstance(entry, dict) else {}

        start = _coerce_positive_float(segment.get("start")) or 0.0
        end_raw = segment.get("end")
        try:
            end = float(end_raw) if end_raw is not None else start
        except (TypeError, ValueError):
            end = start
        duration = _coerce_positive_float(data.get("duration_s"))
        if duration is None:
            duration = max(0.0, end - start)
        data["duration_s"] = duration

        words = _coerce_int(data.get("words"))
        if words is None:
            text = segment.get("text") or ""
            words = len(str(text).split())
        data["words"] = max(0, words)

        pause_time = _coerce_positive_float(data.get("pause_time_s"))
        if pause_time is None:
            pause_time = _coerce_positive_float(data.get("pause_time"))
        if pause_time is None:
            pause_time = 0.0
        data["pause_time_s"] = pause_time

        pause_ratio = _coerce_positive_float(data.get("pause_ratio"))
        if pause_ratio is None:
            pause_ratio = (pause_time / duration) if duration > 0 else 0.0
        data["pause_ratio"] = max(0.0, min(1.0, pause_ratio))

        pause_count = _coerce_int(data.get("pause_count"))
        if pause_count is None:
            pause_count = 0
        data["pause_count"] = max(0, pause_count)

        wpm = _coerce_positive_float(data.get("wpm"))
        if wpm is None:
            if duration > 0:
                wpm = (data["words"] / duration) * 60.0
            else:
                wpm = 0.0
        data["wpm"] = max(0.0, float(wpm))

        normalised[idx] = data

    return normalised


def run(pipeline: AudioAnalysisPipelineV2, state: PipelineState, guard: StageGuard) -> None:
    metrics: dict[int, dict[str, object]] = {}
    state.para_metrics = metrics

    config = pipeline.stats.config_snapshot
    upstream_failed = bool(config.get("transcribe_failed")) or bool(
        config.get("preprocess_failed")
    )
    if upstream_failed:
        guard.progress("skip: upstream stage reported a failure")
        guard.done(count=0)
        return

    if not state.norm_tx:
        guard.progress("skip: no transcript segments available")
        guard.done(count=0)
        return

    audio = state.y
    if isinstance(audio, np.ndarray):
        wav_view = audio
    else:
        wav_view = np.asarray(audio, dtype=np.float32)

    tmp_metrics = pipeline._extract_paraling(wav_view, state.sr, state.norm_tx)
    metrics = _normalise_metrics(state.norm_tx, tmp_metrics if isinstance(tmp_metrics, dict) else {})
    state.para_metrics = metrics

    guard.done(count=len(metrics))
