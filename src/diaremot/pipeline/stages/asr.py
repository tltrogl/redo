"""Automatic speech recognition stage."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import time
from typing import TYPE_CHECKING, Any

from ..logging_utils import StageGuard
from ..pipeline_checkpoint_system import ProcessingStage
from .base import PipelineState
from .utils import atomic_write_json

if TYPE_CHECKING:
    from ..orchestrator import AudioAnalysisPipelineV2

__all__ = ["run"]


def _placeholder_segment(seg: dict[str, Any]) -> Any:
    return type(
        "TranscriptionSegment",
        (),
        {
            "start_time": seg["start_time"],
            "end_time": seg["end_time"],
            "text": "[Transcription unavailable]",
            "speaker_id": seg["speaker_id"],
            "speaker_name": seg["speaker_name"],
            "asr_logprob_avg": None,
            "snr_db": None,
        },
    )()


def _segment_start(item: Any) -> float:
    if hasattr(item, "start_time"):
        start = getattr(item, "start_time")
        if start is not None:
            return float(start)
    if hasattr(item, "start"):
        start = getattr(item, "start")
        if start is not None:
            return float(start)
    if isinstance(item, dict):
        start = item.get("start_time") or item.get("start")
        if start is not None:
            return float(start)
    return 0.0


def _normalize_segment(item: Any) -> dict[str, Any]:
    if hasattr(item, "__dict__"):
        payload = item.__dict__
    elif isinstance(item, dict):
        payload = item
    else:
        payload = {}
    start = float(payload.get("start_time", payload.get("start", 0.0)) or 0.0)
    end = float(payload.get("end_time", payload.get("end", 0.0)) or 0.0)
    return {
        "start": start,
        "end": end,
        "speaker_id": payload.get("speaker_id"),
        "speaker_name": payload.get("speaker_name"),
        "text": payload.get("text", ""),
        "asr_logprob_avg": payload.get("asr_logprob_avg"),
        "snr_db": payload.get("snr_db"),
        "error_flags": payload.get("error_flags", ""),
    }


def _digest_payload(segment: dict[str, Any]) -> dict[str, Any]:
    return {
        "start": float(segment.get("start", 0.0) or 0.0),
        "end": float(segment.get("end", 0.0) or 0.0),
        "speaker_id": segment.get("speaker_id"),
        "speaker_name": segment.get("speaker_name"),
        "text": segment.get("text", ""),
        "asr_logprob_avg": segment.get("asr_logprob_avg"),
        "snr_db": segment.get("snr_db"),
        "error_flags": segment.get("error_flags", ""),
    }


def _segment_digest(segment: dict[str, Any]) -> str:
    payload = json.dumps(_digest_payload(segment), sort_keys=True, ensure_ascii=False)
    return hashlib.blake2s(payload.encode("utf-8"), digest_size=16).hexdigest()


def _cache_metadata(segment: dict[str, Any]) -> dict[str, Any]:
    return {
        "digest": _segment_digest(segment),
        "start": float(segment.get("start", 0.0) or 0.0),
        "end": float(segment.get("end", 0.0) or 0.0),
        "speaker_id": segment.get("speaker_id"),
        "speaker_name": segment.get("speaker_name"),
    }


def _digests_match(
    cached_entries: list[dict[str, Any]],
    fresh_entries: list[dict[str, Any]],
) -> bool:
    if len(cached_entries) != len(fresh_entries):
        return False
    for cached, fresh in zip(cached_entries, fresh_entries, strict=False):
        if cached.get("digest") != fresh.get("digest"):
            return False
        try:
            cached_start = float(cached.get("start", 0.0) or 0.0)
            cached_end = float(cached.get("end", 0.0) or 0.0)
        except (TypeError, ValueError):
            cached_start, cached_end = 0.0, 0.0
        if (abs(cached_start - fresh.get("start", 0.0)) > 1e-3) or (
            abs(cached_end - fresh.get("end", 0.0)) > 1e-3
        ):
            return False
        if cached.get("speaker_id") != fresh.get("speaker_id"):
            return False
        if cached.get("speaker_name") != fresh.get("speaker_name"):
            return False
    return True


def _load_transcription_checkpoint(
    pipeline: AudioAnalysisPipelineV2, state: PipelineState
) -> list[dict[str, Any]] | None:
    checkpoint, _meta = pipeline.checkpoints.load_checkpoint(
        state.input_audio_path,
        ProcessingStage.TRANSCRIPTION,
        file_hash=state.audio_sha16,
    )
    if checkpoint is None:
        return None
    if isinstance(checkpoint, list):
        return [_normalize_segment(segment) for segment in checkpoint]
    if isinstance(checkpoint, dict) and "segments" in checkpoint:
        payload = checkpoint.get("segments")
        if isinstance(payload, list):
            return [_normalize_segment(segment) for segment in payload]
    return None


def run(pipeline: AudioAnalysisPipelineV2, state: PipelineState, guard: StageGuard) -> None:
    if state.resume_tx and state.tx_cache and (state.tx_cache.get("segments") is not None):
        cached_segments = list(state.tx_cache.get("segments", []) or [])
        if cached_segments and isinstance(cached_segments[0], dict) and "digest" in cached_segments[0]:
            checkpoint_segments = _load_transcription_checkpoint(pipeline, state)
            if checkpoint_segments is None:
                pipeline.corelog.stage(
                    "transcribe",
                    "warn",
                    message="[cache] transcription checkpoint missing; re-running ASR",
                )
            else:
                norm_tx = [_normalize_segment(seg) for seg in checkpoint_segments]
                fresh_entries = [_cache_metadata(seg) for seg in norm_tx]
                if _digests_match(cached_segments, fresh_entries):
                    guard.progress(
                        f"resume (tx cache) using {len(cached_segments)} cached segments"
                    )
                    guard.done(segments=len(cached_segments))
                    state.tx_out = list(norm_tx)
                    state.norm_tx = norm_tx
                    pipeline.checkpoints.create_checkpoint(
                        state.input_audio_path,
                        ProcessingStage.TRANSCRIPTION,
                        norm_tx,
                        progress=60.0,
                    )
                    return

                pipeline.corelog.stage(
                    "transcribe",
                    "warn",
                    message="[cache] transcription digest mismatch; re-running ASR",
                )
        else:
            guard.progress(
                f"resume (tx cache) using {len(cached_segments)} cached segments"
            )
            guard.done(segments=len(cached_segments))

            norm_tx = [_normalize_segment(segment) for segment in cached_segments]
            state.tx_out = cached_segments
            state.norm_tx = norm_tx

            pipeline.checkpoints.create_checkpoint(
                state.input_audio_path,
                ProcessingStage.TRANSCRIPTION,
                norm_tx,
                progress=60.0,
            )
            return

    tx_in: list[dict[str, Any]] = []
    for turn in state.turns:
        start = float(turn.get("start", turn.get("start_time", 0.0)) or 0.0)
        end = float(turn.get("end", turn.get("end_time", start + 0.5)) or (start + 0.5))
        speaker_id = str(turn.get("speaker"))
        speaker_name = turn.get("speaker_name") or speaker_id
        tx_in.append(
            {
                "start_time": start,
                "end_time": end,
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
            }
        )

    tx_out: list[Any] = []
    async_enabled = False
    state.ensure_audio()
    try:
        async_enabled = bool(
            getattr(pipeline.pipeline_config, "enable_async_transcription", False)
        )

        if async_enabled and hasattr(pipeline.tx, "transcribe_segments_async"):

            async def _do_async() -> list[Any]:
                return await pipeline.tx.transcribe_segments_async(state.y, state.sr, tx_in)

            try:
                tx_out = asyncio.run(_do_async()) or []
            except RuntimeError:
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    tx_out = loop.run_until_complete(_do_async()) or []
                finally:
                    asyncio.set_event_loop(None)
                    loop.close()
        else:
            tx_out = pipeline.tx.transcribe_segments(state.y, state.sr, tx_in) or []
    except (
        RuntimeError,
        TimeoutError,
        subprocess.CalledProcessError,
    ) as exc:
        pipeline.corelog.stage(
            "transcribe",
            "warn",
            message=(
                "Transcription failed: "
                f"{exc}; generating placeholder segments. Verify faster-whisper setup or tune --asr-segment-timeout."
            ),
        )
        tx_out = [_placeholder_segment(seg) for seg in tx_in]

    tx_out = list(tx_out or [])
    tx_out.sort(key=_segment_start)

    guard.progress(
        f"processed {len(tx_out)} segments" + (" (async)" if async_enabled else "")
    )
    guard.done(segments=len(tx_out))

    norm_tx: list[dict[str, Any]] = [_normalize_segment(item) for item in tx_out]
    for segment in norm_tx:
        segment.setdefault("error_flags", "")

    state.tx_out = tx_out
    state.norm_tx = norm_tx

    pipeline.checkpoints.create_checkpoint(
        state.input_audio_path,
        ProcessingStage.TRANSCRIPTION,
        norm_tx,
        progress=60.0,
    )

    if state.cache_dir:
        try:
            atomic_write_json(
                state.cache_dir / "tx.json",
                {
                    "version": pipeline.cache_version,
                    "audio_sha16": state.audio_sha16,
                    "pp_signature": state.pp_sig,
                    "segment_count": len(norm_tx),
                    "segments": [_cache_metadata(segment) for segment in norm_tx],
                    "saved_at": time.time(),
                },
            )
        except OSError as exc:
            pipeline.corelog.stage(
                "transcribe",
                "warn",
                message=f"[cache] tx.json write failed: {exc}. Check disk space and cache directory permissions.",
            )
