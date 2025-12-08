"""Affect analysis and assembly stage."""

from __future__ import annotations

import json
import math
import os
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from contextlib import nullcontext
import shutil
from typing import TYPE_CHECKING, Any

import numpy as np

from ..logging_utils import StageGuard
from ..outputs import SegmentStreamWriter, ensure_segment_keys
from .base import PipelineState

if TYPE_CHECKING:
    from ..orchestrator import AudioAnalysisPipelineV2

__all__ = ["run", "run_audio_prepass"]


@dataclass(slots=True)
class _SegmentAudioWindow:
    """Lightweight view into a shared waveform buffer.

    The class keeps a cached NumPy slice and reuses a shared ``memoryview`` so
    that downstream analyzers can access contiguous audio without allocating a
    fresh array for every segment. Callers may request either the NumPy view via
    :meth:`as_array` or a zero-copy :class:`memoryview` through :meth:`as_memoryview`.
    """

    _source: np.ndarray
    _memory: memoryview | None
    start: int
    end: int
    _cached: np.ndarray | None = None

    def __post_init__(self) -> None:
        total = int(self._source.shape[0])
        if not (0 <= self.start <= total and 0 <= self.end <= total):
            raise ValueError(f"Invalid range [{self.start}, {self.end}) for array of length {total}")
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) cannot be less than start ({self.start})")

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.end - self.start

    def is_empty(self) -> bool:
        return self.end <= self.start

    def as_array(self, *, dtype: np.dtype | None = None, copy: bool = False) -> np.ndarray:
        """Return a NumPy view of the window, avoiding copies when possible."""

        if dtype is None:
            dtype = np.float32
        if self._cached is None:
            self._cached = self._source[self.start : self.end]

        view = self._cached
        if dtype is not None and view.dtype != dtype:
            view = view.astype(dtype, copy=False)

        if copy:
            return np.array(view, dtype=dtype or view.dtype, copy=True)
        return view

    def as_memoryview(self) -> memoryview | None:
        """Return a zero-copy ``memoryview`` into the underlying audio buffer."""

        if self._memory is None or self.is_empty():
            return None
        return self._memory[self.start : self.end]

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:  # pragma: no cover - exercised implicitly
        return self.as_array(dtype=dtype, copy=False)


class _SegmentAudioFactory:
    """Factory that hands out shared audio windows for affect analysis."""

    __slots__ = ("_source", "_memory")

    def __init__(self, source: np.ndarray) -> None:
        if isinstance(source, np.ndarray):
            self._source = source
        else:  # pragma: no cover - defensive
            self._source = np.asarray(source, dtype=np.float32)
        if self._source.ndim != 1:
            raise ValueError(f"Expected 1D audio array, got shape {self._source.shape}")
        self._memory = memoryview(self._source) if self._source.size else None

    def segment(self, start: int, end: int) -> _SegmentAudioWindow:
        return _SegmentAudioWindow(self._source, self._memory, start, end)


def _estimate_snr_db_from_noise(noise_score: Any) -> float | None:
    """Convert a PANNs noise score into an approximate SNR in dB.

    The raw ``noise_score`` returned by :class:`PANNSEventTagger` is a sum of
    clip-wise probabilities for labels that are considered "noise-like". In
    practice the value tends to fall within ``[0, ~2]`` for speech recordings.

    We map that scalar onto a coarse signal-to-noise ratio estimate using a
    logarithmic curve so that small increases in noise probability have a
    noticeable impact while still saturating gracefully for very noisy clips.
    The heuristic below assumes ~35 dB SNR for pristine audio and rolls off
    toward 0 dB as ``noise_score`` grows. Results are clamped to ``[-5, 35]``
    so downstream consumers always receive a finite float.
    """

    try:
        score = float(noise_score)
    except (TypeError, ValueError):
        return None

    if score <= 0.0:
        return 35.0

    snr = 35.0 - 20.0 * math.log10(1.0 + 10.0 * score)
    if snr < -5.0:
        return -5.0
    if snr > 35.0:
        return 35.0
    return snr


def _load_timeline_events(sed_payload: dict[str, Any]) -> list[Any]:
    """Load timeline events on demand from the SED payload."""

    if not isinstance(sed_payload, dict):
        return []

    direct = sed_payload.get("timeline_events")
    if isinstance(direct, list):
        return direct

    events_path = sed_payload.get("timeline_events_path")
    if not events_path:
        return []

    try:
        path_obj = Path(events_path)
    except (TypeError, ValueError):
        return []

    try:
        with path_obj.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return []

    if isinstance(payload, dict):
        data = payload.get("events")
        if isinstance(data, list):
            return data
    elif isinstance(payload, list):
        return payload

    return []


def _json_dumps_safe(payload: Any, fallback: str) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False)
    except (TypeError, ValueError):
        return fallback


def run_audio_prepass(pipeline: AudioAnalysisPipelineV2, state: PipelineState) -> list[dict[str, Any]]:
    """Optional audio-only affect pass. Does not write primary outputs."""

    audio_rows: list[dict[str, Any]] = []

    sed_payload = state.sed_info or {}
    noise_score = None
    timeline_event_count = None
    timeline_mode = None
    timeline_inference_mode = None
    timeline_events_path = None
    timeline_events: list[Any] = []
    if isinstance(sed_payload, dict):
        noise_score = sed_payload.get("noise_score")
        try:
            noise_score = float(noise_score) if noise_score is not None else None
        except (TypeError, ValueError):
            noise_score = None
        timeline_event_count = sed_payload.get("timeline_event_count")
        try:
            timeline_event_count = int(timeline_event_count) if timeline_event_count is not None else None
        except (TypeError, ValueError):
            timeline_event_count = None
        timeline_mode = sed_payload.get("timeline_mode")
        timeline_inference_mode = sed_payload.get("timeline_inference_mode")
        timeline_events_path = sed_payload.get("timeline_events_path")
        if isinstance(sed_payload.get("timeline_events"), list):
            timeline_events = sed_payload["timeline_events"]
        else:
            path_hint = sed_payload.get("timeline_events_path")
            if path_hint:
                try:
                    count = int(sed_payload.get("timeline_event_count", 0) or 0)
                    should_load = count > 0
                except (TypeError, ValueError):
                    should_load = True
                if should_load:
                    timeline_events = _load_timeline_events(sed_payload)
    timeline_index = _build_timeline_index(timeline_events)

    analyzer = getattr(pipeline, "affect", None)
    if analyzer is None:
        return audio_rows

    audio_buffer = state.ensure_audio()
    audio_windows = _SegmentAudioFactory(audio_buffer)

    for idx, turn in enumerate(state.turns):
        try:
            start = float(turn.get("start", turn.get("start_time", 0.0)) or 0.0)
            end = float(turn.get("end", turn.get("end_time", start)) or start)
        except (TypeError, ValueError):
            start, end = 0.0, 0.0

        i0 = int(max(0.0, start) * state.sr)
        i1 = int(max(0.0, end) * state.sr)
        clip_window = audio_windows.segment(i0, i1)

        try:
            wav = clip_window.as_array(dtype=np.float32, copy=False)
            speech_res = analyzer.analyze_audio(wav, state.sr)
            vad_res = analyzer.analyze_vad_emotion(wav, state.sr)
            speech_emotion = {
                "top": speech_res.top,
                "scores_8class": dict(speech_res.scores),
                "low_confidence_ser": speech_res.low_confidence,
            }
            vad = {
                "valence": vad_res.valence,
                "arousal": vad_res.arousal,
                "dominance": vad_res.dominance,
            }
        except Exception:
            speech_emotion = {"top": "neutral", "scores_8class": {"neutral": 1.0}, "low_confidence_ser": True}
            vad = {"valence": None, "arousal": None, "dominance": None}

        speaker_id = turn.get("speaker_id") or turn.get("speaker") or f"Speaker_{idx+1}"

        row: dict[str, Any] = {
            "file_id": pipeline.stats.file_id,
            "start": start,
            "end": end,
            "speaker_id": speaker_id,
            "speaker_name": turn.get("speaker_name") or speaker_id,
            "text": "",
            "valence": vad.get("valence"),
            "arousal": vad.get("arousal"),
            "dominance": vad.get("dominance"),
            "emotion_top": speech_emotion.get("top", "neutral"),
            "emotion_scores_json": _json_dumps_safe(speech_emotion.get("scores_8class", {"neutral": 1.0}), "{}"),
            "text_emotions_top5_json": "[]",
            "text_emotions_full_json": "{}",
            "intent_top": None,
            "intent_top3_json": "[]",
            "duration_s": max(0.0, end - start),
            "words": 0,
            "pause_ratio": 0.0,
            "vad_unstable": bool(state.vad_unstable),
            "low_confidence_ser": bool(speech_emotion.get("low_confidence_ser", False)),
            "noise_score": noise_score,
            "timeline_event_count": timeline_event_count or 0,
            "timeline_mode": timeline_mode,
            "timeline_inference_mode": timeline_inference_mode,
            "timeline_events_path": timeline_events_path,
            "noise_tag": sed_payload.get("dominant_label") if isinstance(sed_payload, dict) else None,
        }

        row["affect_hint"] = pipeline._affect_hint(
            row.get("valence"), row.get("arousal"), row.get("dominance"), "status_update"
        )

        snr_db_sed = _estimate_snr_db_from_noise(noise_score)
        events_top = []
        if timeline_index is not None:
            overlaps = _intersect_events(start, end, timeline_index)
            if overlaps:
                row["timeline_overlap_count"] = len(overlaps)
                total_overlap = sum(float(item.get("overlap", 0.0)) for item in overlaps)
                if row["duration_s"] > 0:
                    row["timeline_overlap_ratio"] = max(0.0, min(1.0, total_overlap / row["duration_s"]))
                events_top = _topk_by_overlap(overlaps, k=3)
                snr_from_events = _estimate_snr_from_events(overlaps, max(1e-6, row["duration_s"]))
                if snr_from_events is not None:
                    snr_db_sed = snr_from_events
            row["events_top3_json"] = _json_dumps_safe(events_top[:3], "[]")
        elif sed_payload:
            events_top = sed_payload.get("top") or []
            row["events_top3_json"] = _json_dumps_safe(events_top[:3], "[]")

        if snr_db_sed is not None:
            row["snr_db_sed"] = snr_db_sed

        row["_affect_payload"] = {
            "speech_top": speech_emotion.get("top"),
            "speech_scores": speech_emotion.get("scores_8class"),
            "text_full": None,
            "text_top": None,
            "intent_top": None,
            "intent_top3": None,
            "vad": vad,
        }

        finalized = ensure_segment_keys(row)
        audio_rows.append(finalized)
        turn["_audio_affect"] = finalized

    state.audio_affect = audio_rows
    return audio_rows


def run(pipeline: AudioAnalysisPipelineV2, state: PipelineState, guard: StageGuard) -> None:
    segments_final: list[dict[str, Any]] = []

    # Streaming mode is enabled by default to keep peak memory lower. Set
    # DIAREMOT_AFFECT_STREAMING=0/false to disable.
    env_flag = str(os.getenv("DIAREMOT_AFFECT_STREAMING", "")).strip().lower()
    streaming_enabled = env_flag not in {"0", "false", "no", "off"}
    out_dir = Path(getattr(state, "out_dir", Path(".")))
    tmp_out_dir = out_dir / ".affect_tmp"
    writer_cm: SegmentStreamWriter | nullcontext = (
        SegmentStreamWriter(
            tmp_out_dir,
            file_id=pipeline.stats.file_id,
            include_timeline=True,
            include_readable=True,
            mode="w",
        )
        if streaming_enabled
        else nullcontext()
    )

    config = pipeline.stats.config_snapshot
    if bool(config.get("transcribe_failed")) or bool(config.get("preprocess_failed")):
        guard.progress("skip: upstream stage reported a failure")
        state.segments_final = segments_final
        guard.done(segments=0)
        return

    sed_payload = state.sed_info or {}
    noise_score = None
    timeline_event_count = None
    timeline_mode = None
    timeline_inference_mode = None
    timeline_events_path = None
    timeline_events: list[Any] = []
    if isinstance(sed_payload, dict):
        noise_score = sed_payload.get("noise_score")
        if noise_score is not None:
            try:
                noise_score = float(noise_score)
            except (TypeError, ValueError):
                pass
        timeline_event_count = sed_payload.get("timeline_event_count")
        if timeline_event_count is not None:
            try:
                timeline_event_count = int(timeline_event_count)
            except (TypeError, ValueError):
                pass
        timeline_mode = sed_payload.get("timeline_mode")
        if timeline_mode is not None:
            timeline_mode = str(timeline_mode)
        timeline_inference_mode = sed_payload.get("timeline_inference_mode")
        if timeline_inference_mode is not None:
            timeline_inference_mode = str(timeline_inference_mode)
        timeline_events_path = sed_payload.get("timeline_events_path")
        if timeline_events_path is not None:
            timeline_events_path = str(timeline_events_path)
        if isinstance(sed_payload.get("timeline_events"), list):
            timeline_events = sed_payload["timeline_events"]
        else:
            path_hint = sed_payload.get("timeline_events_path")
            if path_hint:
                count = sed_payload.get("timeline_event_count")
                should_load = True
                if count is not None:
                    try:
                        should_load = int(count) > 0
                    except (TypeError, ValueError):
                        should_load = True
                if should_load:
                    timeline_events = _load_timeline_events(sed_payload)
    timeline_index = _build_timeline_index(timeline_events)

    audio_windows = _SegmentAudioFactory(state.y)
    write_errors: list[str] = []

    try:
        with writer_cm as writer:
            for idx, seg in enumerate(state.norm_tx):
                start = float(seg.get("start") or 0.0)
                end = float(seg.get("end") or start)
                i0 = int(start * state.sr)
                i1 = int(end * state.sr)
                clip_window = audio_windows.segment(max(0, i0), max(0, i1))
                text = seg.get("text") or ""

                audio_base = seg.get("_audio_affect")
                aff = None
                vad: dict[str, Any] = {}
                speech_emotion: dict[str, Any] = {}

                if audio_base:
                    payload = audio_base.get("_affect_payload", {})
                    vad = payload.get("vad", {})
                    speech_emotion = {
                        "top": payload.get("speech_top", audio_base.get("emotion_top")),
                        "scores_8class": payload.get("speech_scores"),
                        "low_confidence_ser": audio_base.get("low_confidence_ser"),
                    }

                if not vad or not speech_emotion:
                    aff = pipeline._affect_unified(clip_window, state.sr, text)
                    vad = vad or aff.get("vad", {})
                    speech_emotion = speech_emotion or aff.get("speech_emotion", {})

                # Text affect / intent (avoid rerunning audio if we already have it)
                text_emotions: dict[str, Any] = {}
                intent: dict[str, Any] = {}
                analyzer = getattr(pipeline, "affect", None)
                if analyzer is not None:
                    try:
                        text_res = analyzer.analyze_text(text)
                        text_emotions = {
                            "top5": [dict(item) for item in text_res.top5],
                            "full_28class": dict(text_res.full),
                        }
                    except Exception:
                        text_emotions = {}
                    try:
                        intent_res = analyzer._intent_analyzer.infer(text)
                        intent = {
                            "top": intent_res.top,
                            "top3": [dict(item) for item in intent_res.top3],
                        }
                    except Exception:
                        intent = {}

                if not text_emotions and aff is not None:
                    text_emotions = aff.get("text_emotions", {})
                if not intent and aff is not None:
                    intent = aff.get("intent", {})

                pm = state.para_metrics.get(idx, {})

                tokens_payload = seg.get("asr_tokens")
                if isinstance(tokens_payload, (list, tuple, set)):
                    tokens_payload = list(tokens_payload)
                words_payload = seg.get("asr_words")
                if isinstance(words_payload, (list, tuple, set)):
                    words_payload = list(words_payload)

                if tokens_payload is None:
                    tokens_json = "[]"
                else:
                    try:
                        tokens_json = json.dumps(tokens_payload, ensure_ascii=False)
                    except (TypeError, ValueError):
                        tokens_json = "[]"

                if words_payload is None:
                    words_json = "[]"
                else:
                    try:
                        words_json = json.dumps(words_payload, ensure_ascii=False)
                    except (TypeError, ValueError):
                        words_json = "[]"

                voice_quality_hint = pm.get("vq_note")
                if voice_quality_hint is not None:
                    voice_quality_hint = str(voice_quality_hint)

                duration_s = _coerce_positive_float(pm.get("duration_s"))
                if duration_s is None:
                    duration_s = max(0.0, end - start)

                words = _coerce_int(pm.get("words"))
                if words is None:
                    words = len(text.split())

                pause_ratio = _coerce_positive_float(pm.get("pause_ratio"))
                if pause_ratio is None:
                    pause_time = _coerce_positive_float(pm.get("pause_time_s")) or 0.0
                    pause_ratio = (pause_time / duration_s) if duration_s > 0 else 0.0
                pause_ratio = max(0.0, min(1.0, pause_ratio))

                def _float_or_none(value: Any) -> float | None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return None

                valence = _float_or_none(vad.get("valence")) if vad else None
                arousal = _float_or_none(vad.get("arousal")) if vad else None
                dominance = _float_or_none(vad.get("dominance")) if vad else None

                intent_top = intent.get("top", "status_update") if isinstance(intent, dict) else "status_update"

                row_noise_score = audio_base.get("noise_score") if audio_base else noise_score
                row = {
                    "file_id": pipeline.stats.file_id,
                    "start": start,
                    "end": end,
                    "speaker_id": seg.get("speaker_id"),
                    "speaker_name": seg.get("speaker_name"),
                    "text": text,
                    "valence": valence,
                    "arousal": arousal,
                    "dominance": dominance,
                    "emotion_top": speech_emotion.get("top", "neutral"),
                    "emotion_scores_json": _json_dumps_safe(
                        speech_emotion.get("scores_8class", {"neutral": 1.0}), "{}"
                    ),
                    "text_emotions_top5_json": json.dumps(
                        text_emotions.get("top5", [{"label": "neutral", "score": 1.0}]),
                        ensure_ascii=False,
                    ),
                    "text_emotions_full_json": json.dumps(
                        text_emotions.get("full_28class", {"neutral": 1.0}), ensure_ascii=False
                    ),
                    "intent_top": intent_top,
                    "intent_top3_json": json.dumps(intent.get("top3", []), ensure_ascii=False),
                    "low_confidence_ser": bool(speech_emotion.get("low_confidence_ser", False)),
                    "vad_unstable": bool(state.vad_unstable),
                    "affect_hint": pipeline._affect_hint(valence, arousal, dominance, intent_top),
                    "noise_tag": (audio_base or sed_payload or {}).get("dominant_label") if isinstance(sed_payload, dict) or audio_base else None,
                    "noise_score": row_noise_score,
                    "timeline_event_count": (
                        audio_base.get("timeline_event_count") if audio_base else timeline_event_count
                    ),
                    "timeline_mode": audio_base.get("timeline_mode") if audio_base else timeline_mode,
                    "timeline_inference_mode": (
                        audio_base.get("timeline_inference_mode") if audio_base else timeline_inference_mode
                    ),
                    "timeline_events_path": (
                        audio_base.get("timeline_events_path") if audio_base else timeline_events_path
                    ),
                    "asr_logprob_avg": seg.get("asr_logprob_avg"),
                    "asr_confidence": seg.get("asr_confidence"),
                    "asr_language": seg.get("asr_language"),
                    "asr_tokens_json": tokens_json,
                    "asr_words_json": words_json,
                    "snr_db": seg.get("snr_db"),
                    "wpm": pm.get("wpm", 0.0),
                    "duration_s": duration_s,
                    "words": words,
                    "pause_ratio": pause_ratio,
                    "pause_count": pm.get("pause_count", 0),
                    "pause_time_s": pm.get("pause_time_s", 0.0),
                    "f0_mean_hz": pm.get("f0_mean_hz", 0.0),
                    "f0_std_hz": pm.get("f0_std_hz", 0.0),
                    "loudness_rms": pm.get("loudness_rms", 0.0),
                    "disfluency_count": pm.get("disfluency_count", 0),
                    "vq_jitter_pct": pm.get("vq_jitter_pct"),
                    "vq_shimmer_db": pm.get("vq_shimmer_db"),
                    "vq_hnr_db": pm.get("vq_hnr_db"),
                    "vq_cpps_db": pm.get("vq_cpps_db"),
                    "vq_voiced_ratio": pm.get("vq_voiced_ratio"),
                    "vq_spectral_slope_db": pm.get("vq_spectral_slope_db"),
                    "vq_reliable": bool(pm.get("vq_reliable")),
                    "voice_quality_hint": voice_quality_hint,
                    "error_flags": seg.get("error_flags", ""),
                }

                row["timeline_overlap_count"] = (
                    audio_base.get("timeline_overlap_count", 0) if audio_base else 0
                )
                row["timeline_overlap_ratio"] = (
                    audio_base.get("timeline_overlap_ratio", 0.0) if audio_base else 0.0
                )

                row["_affect_payload"] = {
                    "speech_top": speech_emotion.get("top"),
                    "speech_scores": speech_emotion.get("scores_8class"),
                    "text_full": text_emotions.get("full_28class"),
                    "text_top": text_emotions.get("top5"),
                    "intent_top": intent.get("top"),
                    "intent_top3": intent.get("top3"),
                }

                events_top = []
                snr_db_sed = audio_base.get("snr_db_sed") if audio_base else None
                if audio_base:
                    try:
                        events_top = json.loads(audio_base.get("events_top3_json", "[]"))
                    except Exception:
                        events_top = []
                if not events_top and isinstance(sed_payload, dict) and sed_payload:
                    events_top = sed_payload.get("top") or []
                    row["noise_tag"] = sed_payload.get("dominant_label")
                    snr_db_sed = _estimate_snr_db_from_noise(sed_payload.get("noise_score"))

                if timeline_index is not None:
                    overlaps = _intersect_events(start, end, timeline_index)
                    if overlaps:
                        row["timeline_overlap_count"] = len(overlaps)
                        total_overlap = sum(float(item.get("overlap", 0.0)) for item in overlaps)
                        if duration_s > 0:
                            overlap_ratio = max(0.0, min(1.0, total_overlap / duration_s))
                            row["timeline_overlap_ratio"] = overlap_ratio
                        events_top = _topk_by_overlap(overlaps, k=3)
                        snr_from_events = _estimate_snr_from_events(
                            overlaps, max(1e-6, row.get("duration_s", end - start))
                        )
                        if snr_from_events is not None:
                            snr_db_sed = snr_from_events
                    try:
                        row["events_top3_json"] = json.dumps(events_top[:3], ensure_ascii=False)
                    except (TypeError, ValueError):
                        row["events_top3_json"] = "[]"
                elif events_top:
                    try:
                        row["events_top3_json"] = json.dumps(events_top[:3], ensure_ascii=False)
                    except (TypeError, ValueError):
                        row["events_top3_json"] = "[]"

                if snr_db_sed is not None:
                    row["snr_db_sed"] = snr_db_sed

                finalized = ensure_segment_keys(row)
                segments_final.append(finalized)
                if streaming_enabled and writer is not None:
                    try:
                        writer.write_segment(finalized, index=idx + 1)
                    except Exception as exc:
                        write_errors.append(str(exc))
    except Exception as exc:  # pragma: no cover - ensure partial data persists
        pipeline.corelog.stage(
            "affect_and_assemble",
            "warn",
            message=f"[streaming outputs] failed to write all rows: {exc}",
        )

    if write_errors:
        pipeline.corelog.stage(
            "affect_and_assemble",
            "warn",
            message="; ".join({f"segment write error: {err}" for err in write_errors}),
        )

    # Atomically replace provisional ASR outputs only after affect finished without fatal errors.
    if streaming_enabled and tmp_out_dir.exists():
        try:
            for fname in (
                "diarized_transcript_with_emotion.csv",
                "segments.jsonl",
                "timeline.csv",
                "diarized_transcript_readable.txt",
            ):
                tmp_path = tmp_out_dir / fname
                final_path = out_dir / fname
                if tmp_path.exists():
                    tmp_path.replace(final_path)
        except Exception as exc:
            pipeline.corelog.stage(
                "affect_and_assemble",
                "warn",
                message=f"[streaming outputs] failed to promote affect files: {exc}",
            )
        finally:
            try:
                shutil.rmtree(tmp_out_dir, ignore_errors=True)
            except Exception:
                pass

    state.segments_final = segments_final

    if timeline_index is not None and isinstance(state.turns, list):
        for turn in state.turns:
            if not isinstance(turn, dict):
                continue
            try:
                start = float(turn.get("start", turn.get("start_time", 0.0)) or 0.0)
                end = float(turn.get("end", turn.get("end_time", start)) or start)
            except (TypeError, ValueError):
                continue
            overlaps = _intersect_events(start, end, timeline_index)
            if overlaps:
                turn["events_top3"] = _topk_by_overlap(overlaps, k=3)
                snr_turn = _estimate_snr_from_events(overlaps, max(1e-6, end - start))
                if snr_turn is not None:
                    turn["snr_db_sed"] = snr_turn
                turn["timeline_overlap_count"] = len(overlaps)
                total_overlap = sum(float(item.get("overlap", 0.0)) for item in overlaps)
                duration = max(1e-6, end - start)
                turn["timeline_overlap_ratio"] = max(0.0, min(1.0, total_overlap / duration))
            elif "events_top3" not in turn:
                turn["events_top3"] = []
                turn.setdefault("timeline_overlap_count", 0)
                turn.setdefault("timeline_overlap_ratio", 0.0)

    guard.done(segments=len(segments_final))


def _coerce_positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result < 0:
        return None
    return result


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _build_timeline_index(events: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not events:
        return None
    try:
        sorted_events = sorted(events, key=lambda ev: float(ev.get("start", 0.0)))
    except Exception:
        return None
    starts = [float(ev.get("start", 0.0)) for ev in sorted_events]
    return {"events": sorted_events, "starts": starts}


def _intersect_events(start: float, end: float, index: dict[str, Any]) -> list[dict[str, Any]]:
    events = index.get("events") or []
    starts = index.get("starts") or []
    if not events or not starts or end <= start:
        return []
    overlaps: list[dict[str, Any]] = []
    idx = max(0, bisect_left(starts, start) - 1)
    while idx < len(events):
        event = events[idx]
        ev_start = float(event.get("start", 0.0))
        ev_end = float(event.get("end", ev_start))
        if ev_start >= end:
            break
        if ev_end <= start:
            idx += 1
            continue
        overlap_start = max(start, ev_start)
        overlap_end = min(end, ev_end)
        overlap = overlap_end - overlap_start
        if overlap > 0:
            score = float(event.get("score", 0.0))
            overlaps.append(
                {
                    "label": event.get("label"),
                    "score": score,
                    "start": ev_start,
                    "end": ev_end,
                    "overlap": overlap,
                    "weight": overlap * max(0.0, score),
                }
            )
        idx += 1
    return overlaps


def _topk_by_overlap(overlaps: list[dict[str, Any]], *, k: int) -> list[dict[str, Any]]:
    if not overlaps:
        return []
    aggregates: dict[str, dict[str, float | str]] = {}
    for item in overlaps:
        label = str(item.get("label", "unknown"))
        slot = aggregates.setdefault(label, {"label": label, "overlap": 0.0, "weight": 0.0, "score": 0.0})
        overlap = float(item.get("overlap", 0.0))
        weight = float(item.get("weight", 0.0))
        slot["overlap"] = float(slot["overlap"]) + overlap
        slot["weight"] = float(slot["weight"]) + weight
    for slot in aggregates.values():
        overlap = float(slot.get("overlap", 0.0))
        weight = float(slot.get("weight", 0.0))
        slot["score"] = (weight / overlap) if overlap > 0 else 0.0
    ranked = sorted(
        aggregates.values(),
        key=lambda item: (float(item.get("weight", 0.0)), float(item.get("score", 0.0))),
        reverse=True,
    )
    return [
        {
            "label": entry.get("label"),
            "score": float(entry.get("score", 0.0)),
            "overlap": float(entry.get("overlap", 0.0)),
        }
        for entry in ranked[:k]
    ]


def _estimate_snr_from_events(overlaps: list[dict[str, Any]], duration: float) -> float | None:
    if not overlaps or duration <= 0:
        return None
    total_overlap = sum(float(item.get("overlap", 0.0)) for item in overlaps)
    total_weight = sum(float(item.get("weight", 0.0)) for item in overlaps)
    if total_overlap <= 0:
        return None
    density = min(1.0, total_overlap / duration)
    weighted = min(1.0, total_weight / duration)
    composite = min(1.0, 0.5 * density + 0.5 * weighted)
    snr = 35.0 - 35.0 * composite
    if snr < -5.0:
        return -5.0
    if snr > 35.0:
        return 35.0
    return snr
