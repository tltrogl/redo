"""Paralinguistics extraction stage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..logging_utils import StageGuard
from .base import PipelineState

if TYPE_CHECKING:
    from ..orchestrator import AudioAnalysisPipelineV2

__all__ = ["run"]


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

    state.ensure_audio()
    audio = state.y
    if isinstance(audio, np.ndarray):
        wav_view = audio
    else:
        wav_view = np.asarray(audio, dtype=np.float32)

    tmp_metrics = pipeline._extract_paraling(wav_view, state.sr, state.norm_tx)
    if isinstance(tmp_metrics, dict):
        metrics = tmp_metrics
        state.para_metrics = metrics

    guard.done(count=len(metrics))
