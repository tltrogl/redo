"""Summary and reporting stages."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...summaries.conversation_analysis import (
    ConversationAnalysisError,
    ConversationMetrics,
    analyze_conversation_flow,
)
from ...summaries.speakers_summary_builder import build_speakers_summary
from ..logging_utils import StageGuard
from .base import PipelineState

if TYPE_CHECKING:
    from ..orchestrator import AudioAnalysisPipelineV2

__all__ = [
    "run_overlap",
    "run_conversation",
    "run_speaker_rollups",
    "run_outputs",
]


def run_overlap(pipeline: AudioAnalysisPipelineV2, state: PipelineState, guard: StageGuard) -> None:
    overlap_stats = {"overlap_total_sec": 0.0, "overlap_ratio": 0.0}
    per_speaker = {}
    events: list[dict[str, str | float | int]] = []
    try:
        module = getattr(pipeline, "paralinguistics_module", None)
        compute_overlap = getattr(module, "compute_overlap_and_interruptions")
        overlap = compute_overlap(state.turns) or {}
        overlap_stats = {
            "overlap_total_sec": float(overlap.get("overlap_total_sec", 0.0)),
            "overlap_ratio": float(overlap.get("overlap_ratio", 0.0)),
        }

        per_speaker_map: dict[str, dict[str, float | int]] = {}
        made_provided: set[str] = set()

        def _ensure_entry(key: str) -> dict[str, float | int]:
            return per_speaker_map.setdefault(key, {"made": 0, "received": 0, "overlap_sec": 0.0})

        for speaker_id, values in (overlap.get("by_speaker") or {}).items():
            if not isinstance(values, dict):
                continue
            key = str(speaker_id)
            slot = _ensure_entry(key)
            slot["overlap_sec"] = float(values.get("overlap_sec", 0.0) or 0.0)

            made_raw = values.get("made", values.get("interruptions"))
            try:
                made_val = int(float(made_raw))
            except (TypeError, ValueError):
                made_val = 0
            slot["made"] = made_val
            made_provided.add(key)

        raw_events = overlap.get("interruptions") or []
        if isinstance(raw_events, list):
            for index, item in enumerate(raw_events, start=1):
                if not isinstance(item, dict):
                    continue

                at_raw = item.get("at", 0.0)
                try:
                    at_val = float(at_raw)
                except (TypeError, ValueError):
                    at_val = 0.0

                overlap_raw = item.get("overlap_sec", 0.0)
                try:
                    overlap_val = float(overlap_raw)
                except (TypeError, ValueError):
                    overlap_val = 0.0

                interrupter = item.get("interrupter")
                interrupted = item.get("interrupted")

                event_record: dict[str, str | float | int] = {
                    "index": index,
                    "at": at_val,
                    "overlap_sec": overlap_val,
                    "interrupter": "" if interrupter in (None, "") else str(interrupter),
                    "interrupted": "" if interrupted in (None, "") else str(interrupted),
                }
                events.append(event_record)

                if interrupter not in (None, ""):
                    key = str(interrupter)
                    slot = _ensure_entry(key)
                    if key not in made_provided:
                        slot["made"] = int(slot.get("made", 0)) + 1

                if interrupted in (None, ""):
                    continue
                slot = _ensure_entry(str(interrupted))
                slot["received"] = int(slot.get("received", 0)) + 1

        per_speaker = per_speaker_map
    except (AttributeError, RuntimeError, ValueError) as exc:
        pipeline.corelog.stage(
            "overlap_interruptions",
            "warn",
            message=f"skipped: {exc}. Install paralinguistics extras or validate overlap feature inputs.",
        )
        events = []
    state.overlap_stats = overlap_stats
    state.per_speaker_interrupts = per_speaker
    state.interruption_events = events
    guard.done()


def run_conversation(
    pipeline: AudioAnalysisPipelineV2, state: PipelineState, guard: StageGuard
) -> None:
    try:
        metrics = analyze_conversation_flow(state.segments_final, state.duration_s)
        pipeline.corelog.event(
            "conversation_analysis",
            "metrics",
            balance=metrics.turn_taking_balance,
            pace=metrics.conversation_pace_turns_per_min,
            coherence=metrics.topic_coherence_score,
        )
        state.conv_metrics = metrics
    except (
        ConversationAnalysisError,
        RuntimeError,
        ValueError,
        ZeroDivisionError,
    ) as exc:
        pipeline.corelog.stage(
            "conversation_analysis",
            "warn",
            message=f"analysis failed: {exc}. Falling back to neutral conversational metrics.",
        )
        try:
            state.conv_metrics = ConversationMetrics(
                turn_taking_balance=0.5,
                interruption_rate_per_min=0.0,
                avg_turn_duration_sec=0.0,
                conversation_pace_turns_per_min=0.0,
                silence_ratio=0.0,
                speaker_dominance={},
                response_latency_stats={},
                topic_coherence_score=0.0,
                energy_flow=[],
            )
        except (TypeError, ValueError):
            state.conv_metrics = None
    guard.done()


def run_speaker_rollups(
    pipeline: AudioAnalysisPipelineV2, state: PipelineState, guard: StageGuard
) -> None:
    try:
        summary = build_speakers_summary(
            state.segments_final, state.per_speaker_interrupts
        )
        if isinstance(summary, dict):
            summary = [dict(v, speaker_id=k) for k, v in summary.items()]
        elif not isinstance(summary, list):
            summary = []
        state.speakers_summary = summary
    except (RuntimeError, ValueError, TypeError) as exc:
        pipeline.corelog.stage(
            "speaker_rollups",
            "warn",
            message=f"failed: {exc}. Inspect segment records or disable speaker summary generation.",
        )
        state.speakers_summary = []
    guard.done(count=len(state.speakers_summary))


def run_outputs(pipeline: AudioAnalysisPipelineV2, state: PipelineState, guard: StageGuard) -> None:
    pipeline._write_outputs(
        state.input_audio_path,
        state.out_dir,
        state.segments_final,
        state.speakers_summary,
        state.health,
        state.turns,
        state.overlap_stats,
        state.per_speaker_interrupts,
        state.interruption_events,
        state.conv_metrics,
        state.duration_s,
        state.sed_info,
    )

    if state.cache_dir:
        try:
            (state.cache_dir / ".done").write_text("ok", encoding="utf-8")
        except OSError:
            pass

    guard.done()
