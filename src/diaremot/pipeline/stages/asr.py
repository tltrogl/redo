"""Automatic speech recognition stage."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import time
from typing import TYPE_CHECKING, Any

from ..logging_utils import StageGuard
from ..outputs import SegmentStreamWriter, ensure_segment_keys
from ..pipeline_checkpoint_system import ProcessingStage
from .base import PipelineState
from .utils import atomic_write_json, build_cache_payload

if TYPE_CHECKING:
    from ..orchestrator import AudioAnalysisPipelineV2

__all__ = ["run"]


def _lookup_audio_affect(audio_rows: list[dict[str, Any]], start: float, *, index: int) -> dict[str, Any] | None:
    if index < len(audio_rows):
        return audio_rows[index]
    best: tuple[float, dict[str, Any]] | None = None
    for row in audio_rows:
        try:
            rs = float(row.get("start", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        delta = abs(rs - start)
        if best is None or delta < best[0]:
            best = (delta, row)
    if best and best[0] <= 0.75:
        return best[1]
    return None


def _merge_audio_affect(norm_tx: list[dict[str, Any]], audio_rows: list[dict[str, Any]]) -> None:
    if not audio_rows:
        return
    for i, seg in enumerate(norm_tx):
        start = float(seg.get("start", 0.0) or 0.0)
        audio_row = _lookup_audio_affect(audio_rows, start, index=i)
        if audio_row:
            seg["_audio_affect"] = audio_row


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

    def _normalize_sequence(raw: Any) -> Any:
        if raw is None:
            return None
        if isinstance(raw, (list, tuple, set)):
            normalized: list[Any] = []
            for item in raw:
                if isinstance(item, (str, int, float, bool)) or item is None:
                    normalized.append(item)
                elif hasattr(item, "_asdict"):
                    normalized.append(dict(item._asdict()))
                elif hasattr(item, "__dict__"):
                    normalized.append(
                        {
                            key: value
                            for key, value in item.__dict__.items()
                            if not key.startswith("_")
                        }
                    )
                else:
                    normalized.append(str(item))
            return normalized
        try:
            return list(raw)
        except TypeError:
            return raw

    return {
        "start": start,
        "end": end,
        "speaker_id": payload.get("speaker_id"),
        "speaker_name": payload.get("speaker_name"),
        "text": payload.get("text", ""),
        "asr_logprob_avg": payload.get("asr_logprob_avg"),
        "snr_db": payload.get("snr_db"),
        "error_flags": payload.get("error_flags", ""),
        "asr_confidence": payload.get("confidence"),
        "asr_language": payload.get("language"),
        "asr_tokens": _normalize_sequence(payload.get("tokens")),
        "asr_words": _normalize_sequence(payload.get("words")),
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
        "asr_confidence": segment.get("asr_confidence"),
        "asr_language": segment.get("asr_language"),
        "asr_tokens": segment.get("asr_tokens"),
        "asr_words": segment.get("asr_words"),
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


def _provisional_row(segment: dict[str, Any], file_id: str, audio_row: dict[str, Any] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "file_id": file_id,
        "start": segment.get("start"),
        "end": segment.get("end"),
        "speaker_id": segment.get("speaker_id"),
        "speaker_name": segment.get("speaker_name"),
        "text": segment.get("text", ""),
        "asr_logprob_avg": segment.get("asr_logprob_avg"),
        "snr_db": segment.get("snr_db"),
        "error_flags": segment.get("error_flags", ""),
        "asr_confidence": segment.get("asr_confidence"),
        "asr_language": segment.get("asr_language"),
    }

    try:
        row["asr_tokens_json"] = json.dumps(segment.get("asr_tokens"), ensure_ascii=False)
    except (TypeError, ValueError):
        row["asr_tokens_json"] = "[]"

    try:
        row["asr_words_json"] = json.dumps(segment.get("asr_words"), ensure_ascii=False)
    except (TypeError, ValueError):
        row["asr_words_json"] = "[]"

    if audio_row:
        for key in (
            "valence",
            "arousal",
            "dominance",
            "emotion_top",
            "emotion_scores_json",
            "text_emotions_top5_json",
            "text_emotions_full_json",
            "intent_top",
            "intent_top3_json",
            "noise_score",
            "timeline_event_count",
            "timeline_mode",
            "timeline_inference_mode",
            "timeline_events_path",
            "timeline_overlap_count",
            "timeline_overlap_ratio",
            "events_top3_json",
            "snr_db_sed",
            "noise_tag",
        ):
            value = audio_row.get(key)
            if value not in (None, ""):
                row[key] = value
        row["low_confidence_ser"] = audio_row.get("low_confidence_ser", row.get("low_confidence_ser", False))
        row["vad_unstable"] = audio_row.get("vad_unstable", row.get("vad_unstable", False))
        if "affect_hint" in audio_row:
            row["affect_hint"] = audio_row.get("affect_hint")

    return ensure_segment_keys(row)


def _write_provisional_outputs(
    pipeline: "AudioAnalysisPipelineV2", state: PipelineState
) -> None:
    try:
        with SegmentStreamWriter(
            state.out_dir,
            file_id=pipeline.stats.file_id,
            include_timeline=True,
            include_readable=True,
            mode="w",
        ) as writer:
            for idx, segment in enumerate(state.norm_tx, start=1):
                writer.write_segment(
                    _provisional_row(segment, pipeline.stats.file_id, segment.get("_audio_affect")),
                    index=idx,
                )
    except Exception as exc:  # pragma: no cover - best effort persistence
        pipeline.corelog.stage(
            "transcribe",
            "warn",
            message=(
                "[provisional outputs] failed to persist early transcript: "
                f"{exc}. Verify output directory permissions and disk space."
            ),
        )


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
    prepass_on = (
        os.getenv("DIAREMOT_AUDIO_AFFECT_PREPASS", "")
        .strip()
        .lower()
        not in {"", "0", "false", "no", "off"}
    )
    if prepass_on and not state.audio_affect:
        try:
            from . import affect as affect_stage

            state.audio_affect = affect_stage.run_audio_prepass(pipeline, state) or []
        except Exception as exc:  # pragma: no cover - optional path
            pipeline.corelog.stage(
                "transcribe",
                "warn",
                message=f"[audio prepass] skipped: {exc}",
            )

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
                    _merge_audio_affect(norm_tx, state.audio_affect)
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
            _merge_audio_affect(norm_tx, state.audio_affect)
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

    _merge_audio_affect(norm_tx, state.audio_affect)
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
            payload = build_cache_payload(
                version=pipeline.cache_version,
                audio_sha16=state.audio_sha16,
                pp_signature=state.pp_sig,
                extra={
                    "segment_count": len(norm_tx),
                    "segments": [_cache_metadata(segment) for segment in norm_tx],
                    "saved_at": time.time(),
                },
            )
            atomic_write_json(state.cache_dir / "tx.json", payload)
        except OSError as exc:
            pipeline.corelog.stage(
                "transcribe",
                "warn",
                message=f"[cache] tx.json write failed: {exc}. Check disk space and cache directory permissions.",
            )

    _write_provisional_outputs(pipeline, state)
