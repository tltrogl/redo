"""Output helpers for the DiaRemot pipeline."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

from .logging_utils import RunStats, _make_json_safe

SEGMENT_COLUMNS = [
    "file_id",
    "start",
    "end",
    "speaker_id",
    "speaker_name",
    "text",
    "valence",
    "arousal",
    "dominance",
    "emotion_top",
    "emotion_scores_json",
    "text_emotions_top5_json",
    "text_emotions_full_json",
    "intent_top",
    "intent_top3_json",
    "events_top3_json",
    "noise_tag",
    "asr_logprob_avg",
    "snr_db",
    "snr_db_sed",
    "wpm",
    "duration_s",
    "words",
    "pause_ratio",
    "low_confidence_ser",
    "vad_unstable",
    "affect_hint",
    "pause_count",
    "pause_time_s",
    "f0_mean_hz",
    "f0_std_hz",
    "loudness_rms",
    "disfluency_count",
    "error_flags",
    "vq_jitter_pct",
    "vq_shimmer_db",
    "vq_hnr_db",
    "vq_cpps_db",
    "voice_quality_hint",
]


def default_affect() -> dict[str, Any]:
    ser_scores = {"neutral": 1.0}
    text_full = {"neutral": 1.0}
    return {
        "vad": {"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
        "speech_emotion": {
            "top": "neutral",
            "scores_8class": ser_scores,
            "low_confidence_ser": True,
        },
        "text_emotions": {
            "top5": [{"label": "neutral", "score": 1.0}],
            "full_28class": text_full,
        },
        "intent": {
            "top": "status_update",
            "top3": [
                {"label": "status_update", "score": 1.0},
                {"label": "small_talk", "score": 0.0},
                {"label": "opinion", "score": 0.0},
            ],
        },
        "affect_hint": "neutral-status",
    }


def ensure_segment_keys(seg: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "events_top3_json": "[]",
        "low_confidence_ser": False,
        "vad_unstable": False,
        "error_flags": "",
    }
    for key in SEGMENT_COLUMNS:
        if key not in seg:
            seg[key] = defaults.get(key, None)
    return seg


def write_segments_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEGMENT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, None) for key in SEGMENT_COLUMNS})


def write_segments_jsonl(path: Path, segments: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for segment in segments:
            clean = {
                key: value
                for key, value in (segment or {}).items()
                if not (isinstance(key, str) and key.startswith("_"))
            }
            handle.write(json.dumps(clean, ensure_ascii=False) + "\n")


def write_timeline_csv(path: Path, segments: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["start", "end", "speaker_id"])
        for segment in segments:
            writer.writerow(
                [
                    segment.get("start", 0.0),
                    segment.get("end", 0.0),
                    segment.get("speaker_id", ""),
                ]
            )


def write_narrative_report(path: Path, narrative: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    summary = (narrative or {}).get("summary") or "Conversation summary unavailable."
    emotion_brief = (narrative or {}).get("emotion_brief") or "Emotion analysis unavailable."
    insights = (narrative or {}).get("interaction_insights") or []
    top_emotions = (narrative or {}).get("top_emotions") or []
    emotion_stats = (narrative or {}).get("emotion_stats") or {}
    dominant = (narrative or {}).get("dominant_speaker") or {}

    lines = [
        "# Conversation Overview",
        "",
        summary,
        "",
        "## Emotion Brief",
        "",
        emotion_brief,
        "",
    ]

    if top_emotions:
        lines.extend(["### Top Emotions", ""])
        for label, count in top_emotions:
            lines.append(f"- {label}: {count}")
        lines.append("")

    if emotion_stats:
        metrics_lines = [
            f"- {key.replace('_', ' ').title()}: {value:.2f}"
            for key, value in emotion_stats.items()
            if value is not None
        ]
        if metrics_lines:
            lines.extend(["### Affect Metrics", ""])
            lines.extend(metrics_lines)
            lines.append("")

    dom_name = dominant.get("name")
    dom_pct = dominant.get("percent")
    if dom_name and dom_pct is not None:
        lines.extend(
            [
                "### Dominant Speaker",
                "",
                f"{dom_name} controlled approximately {dom_pct:.0f}% of the speaking time.",
                "",
            ]
        )

    lines.extend(["## Interaction Insights", ""])
    if insights:
        lines.extend(f"- {item}" for item in insights)
    else:
        lines.append("- No notable interaction dynamics detected.")

    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_qc_report(
    path: Path,
    stats: RunStats,
    health: Any,
    *,
    n_turns: int,
    n_segments: int,
    segments: list[dict[str, Any]],
) -> None:
    payload = {
        "run_id": stats.run_id,
        "file_id": stats.file_id,
        "schema_version": stats.schema_version,
        "stage_timings_ms": stats.stage_timings_ms,
        "stage_counts": stats.stage_counts,
        "warnings": stats.warnings,
        "errors": getattr(stats, "errors", []),
        "failures": getattr(stats, "failures", []),
        "models": stats.models,
        "config_snapshot": stats.config_snapshot,
        "audio_health": {
            "snr_db": float(getattr(health, "snr_db", 0.0)) if health else None,
            "silence_ratio": float(getattr(health, "silence_ratio", 0.0)) if health else None,
            "clipping_detected": (
                bool(getattr(health, "clipping_detected", False)) if health else None
            ),
            "dynamic_range_db": float(getattr(health, "dynamic_range_db", 0.0)) if health else None,
        },
        "counts": {"turns": int(n_turns), "segments": int(n_segments)},
    }

    try:

        def _avg(key: str) -> float | None:
            values = []
            for seg in segments or []:
                value = seg.get(key)
                if value is None:
                    continue
                try:
                    values.append(float(value))
                except Exception:
                    continue
            return float(sum(values) / len(values)) if values else None

        payload["voice_quality_summary"] = {
            "vq_jitter_pct_avg": _avg("vq_jitter_pct"),
            "vq_shimmer_db_avg": _avg("vq_shimmer_db"),
            "vq_hnr_db_avg": _avg("vq_hnr_db"),
            "vq_cpps_db_avg": _avg("vq_cpps_db"),
        }
    except Exception:  # pragma: no cover - best effort aggregate
        pass

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_make_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _format_hms(seconds: Any) -> str:
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "--:--:--"
    if value < 0:
        value = 0.0
    total_seconds = int(round(value))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _parse_json_blob(blob: Any) -> Any:
    if isinstance(blob, (dict, list)):
        return blob
    if isinstance(blob, str) and blob.strip():
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            return None
    return None


def _format_float(value: Any, *, signed: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    fmt = "{:+.2f}" if signed else "{:.2f}"
    return fmt.format(number)


def write_human_transcript(path: Path, segments: list[dict[str, Any]]) -> None:
    """Render a human-friendly transcript with diarization context."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    if not segments:
        lines.append("No speech segments detected.")
    else:
        for index, segment in enumerate(segments, start=1):
            start = _format_hms(segment.get("start"))
            end = _format_hms(segment.get("end"))
            speaker = (
                str(segment.get("speaker_name"))
                if segment.get("speaker_name")
                else str(segment.get("speaker_id") or f"Speaker_{index:02d}")
            )
            lines.append(f"[{start} - {end}] {speaker}")

            text = (segment.get("text") or "").strip()
            lines.append(f"  Text: {text or '(no speech recognized)'}")

            valence = _format_float(segment.get("valence"), signed=True)
            arousal = _format_float(segment.get("arousal"), signed=True)
            dominance = _format_float(segment.get("dominance"), signed=True)
            emotion = segment.get("emotion_top") or "unknown"
            affect_hint = segment.get("affect_hint") or "n/a"
            lines.append(
                f"  Affect: emotion {emotion}; valence {valence}, "
                f"arousal {arousal}, dominance {dominance}; hint {affect_hint}"
            )

            intent = segment.get("intent_top") or "unknown"
            intents_blob = _parse_json_blob(segment.get("intent_top3_json"))
            if isinstance(intents_blob, list):
                intent_detail = ", ".join(
                    f"{item.get('label', 'unknown')} {float(item.get('score', 0.0)):.2f}"
                    for item in intents_blob[:3]
                    if isinstance(item, dict)
                )
            else:
                intent_detail = "n/a"
            lines.append(f"  Intent: {intent} (top3: {intent_detail})")

            sed_blob = _parse_json_blob(segment.get("events_top3_json"))
            sed_labels = (
                [item.get("label", "unknown") for item in sed_blob if isinstance(item, dict)]
                if isinstance(sed_blob, list)
                else []
            )
            sed_summary = ", ".join(sed_labels) if sed_labels else "n/a"
            noise = segment.get("noise_tag") or "n/a"
            vad_status = "unstable" if segment.get("vad_unstable") else "stable"
            lines.append(f"  VAD: {vad_status} | SED: {sed_summary} | Noise tag: {noise}")

            duration = _format_float(segment.get("duration_s"))
            wpm = _format_float(segment.get("wpm"))
            lines.append(f"  Duration: {duration}s | Speech rate: {wpm} wpm")
            lines.append("")  # blank line between segments

    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_speakers_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    headers = sorted({key for row in rows for key in row.keys()})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, None) for key in headers})


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        try:
            return json.dumps(_make_json_safe(value), ensure_ascii=False, sort_keys=True)
        except Exception:
            return "{}"


def write_conversation_metrics_csv(
    path: Path,
    metrics: Any,
    *,
    file_id: str | None = None,
    duration_s: float | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "file_id",
        "turn_taking_balance",
        "interruption_rate_per_min",
        "avg_turn_duration_sec",
        "conversation_pace_turns_per_min",
        "silence_ratio",
        "topic_coherence_score",
        "duration_s",
        "speaker_dominance_json",
        "response_latency_stats_json",
        "energy_flow_json",
        "interruptions_per_speaker_json",
    ]

    row: dict[str, Any] = {key: None for key in headers}
    row["file_id"] = file_id or ""
    row["duration_s"] = duration_s
    row["speaker_dominance_json"] = "{}"
    row["response_latency_stats_json"] = "{}"
    row["energy_flow_json"] = "[]"
    row["interruptions_per_speaker_json"] = "{}"

    if metrics is not None:
        if is_dataclass(metrics):
            payload = asdict(metrics)
        elif isinstance(metrics, dict):
            payload = dict(metrics)
        else:
            payload = {
                key: attr
                for key in dir(metrics)
                if not key.startswith("_") and not callable((attr := getattr(metrics, key)))
            }

        for key in [
            "turn_taking_balance",
            "interruption_rate_per_min",
            "avg_turn_duration_sec",
            "conversation_pace_turns_per_min",
            "silence_ratio",
            "topic_coherence_score",
        ]:
            if key in payload:
                row[key] = payload[key]

        row["speaker_dominance_json"] = _safe_json(payload.get("speaker_dominance"))
        row["response_latency_stats_json"] = _safe_json(
            payload.get("response_latency_stats")
        )
        row["energy_flow_json"] = _safe_json(payload.get("energy_flow"))
        row["interruptions_per_speaker_json"] = _safe_json(
            payload.get("interruptions_per_speaker")
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerow(row)


def write_overlap_summary_csv(
    path: Path,
    overlap_stats: dict[str, Any] | None,
    *,
    file_id: str | None = None,
    duration_s: float | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    available = bool(overlap_stats.get("available")) if overlap_stats else False
    total_sec = float(overlap_stats.get("overlap_total_sec", 0.0)) if overlap_stats else 0.0
    ratio = float(overlap_stats.get("overlap_ratio", 0.0)) if overlap_stats else 0.0
    ratio_pct = ratio * 100.0
    normalized_ratio = None
    if duration_s and duration_s > 0:
        normalized_ratio = min(100.0, max(0.0, 100.0 * total_sec / duration_s))

    headers = [
        "file_id",
        "overlap_total_sec",
        "overlap_ratio",
        "overlap_ratio_pct",
        "conversation_duration_sec",
        "overlap_vs_duration_pct",
        "overlap_available",
    ]
    row = {
        "file_id": file_id or "",
        "overlap_total_sec": total_sec,
        "overlap_ratio": ratio,
        "overlap_ratio_pct": ratio_pct,
        "conversation_duration_sec": duration_s,
        "overlap_vs_duration_pct": normalized_ratio,
        "overlap_available": available,
    }

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerow(row)


def write_interruptions_csv(
    path: Path,
    per_speaker: dict[str, Any] | None,
    *,
    speaker_labels: dict[str, str] | None = None,
    conv_metrics: Any = None,
    file_id: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rate_map: dict[str, Any] = {}
    if conv_metrics is not None:
        rate_payload: Any
        if is_dataclass(conv_metrics):
            rate_payload = asdict(conv_metrics).get("interruptions_per_speaker", {})
        elif isinstance(conv_metrics, dict):
            rate_payload = conv_metrics.get("interruptions_per_speaker", {})
        else:
            rate_payload = getattr(conv_metrics, "interruptions_per_speaker", {})
        if isinstance(rate_payload, dict):
            rate_map = {str(k): v for k, v in rate_payload.items()}

    headers = [
        "file_id",
        "speaker_id",
        "speaker_name",
        "interruptions_made",
        "interruptions_received",
        "overlap_sec",
        "interruptions_per_min",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()

        if not per_speaker:
            return

        for speaker_id in sorted(per_speaker.keys(), key=str):
            stats = per_speaker.get(speaker_id) or {}
            sid = str(speaker_id)
            row = {
                "file_id": file_id or "",
                "speaker_id": sid,
                "speaker_name": (speaker_labels or {}).get(sid, ""),
                "interruptions_made": stats.get("made", 0),
                "interruptions_received": stats.get("received", 0),
                "overlap_sec": stats.get("overlap_sec", 0.0),
                "interruptions_per_min": rate_map.get(sid),
            }
            writer.writerow(row)


def write_audio_health_csv(
    path: Path,
    health: Any,
    *,
    file_id: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "file_id",
        "snr_db",
        "dynamic_range_db",
        "silence_ratio",
        "clipping_detected",
        "rms_db",
        "est_lufs",
        "floor_clipping_ratio",
        "is_chunked",
    ]

    row = {key: None for key in headers}
    row["file_id"] = file_id or ""

    if health is not None:
        for key in [
            "snr_db",
            "dynamic_range_db",
            "silence_ratio",
            "clipping_detected",
            "rms_db",
            "est_lufs",
            "floor_clipping_ratio",
            "is_chunked",
        ]:
            if hasattr(health, key):
                row[key] = getattr(health, key)
            elif isinstance(health, dict) and key in health:
                row[key] = health[key]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerow(row)


def write_background_sed_summary_csv(
    path: Path,
    sed_info: dict[str, Any] | None,
    *,
    file_id: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "file_id",
        "enabled",
        "noise_score",
        "dominant_label",
        "top_label",
        "top_label_score",
        "top_labels_json",
        "timeline_event_count",
        "timeline_csv",
        "timeline_jsonl",
        "timeline_mode",
        "timeline_inference_mode",
        "timeline_events_path",
    ]

    row: dict[str, Any] = {key: None for key in headers}
    row["file_id"] = file_id or ""

    if sed_info:
        row["enabled"] = sed_info.get("enabled")
        row["noise_score"] = sed_info.get("noise_score")
        row["dominant_label"] = sed_info.get("dominant_label")
        top = sed_info.get("top") or []
        top_label = None
        top_score = None
        if isinstance(top, list) and top:
            candidate = top[0]
            if isinstance(candidate, dict):
                top_label = candidate.get("label") or candidate.get("name")
                top_score = candidate.get("score") or candidate.get("weight")
            elif isinstance(candidate, (list, tuple)) and candidate:
                top_label = candidate[0]
                top_score = candidate[1] if len(candidate) > 1 else None
        row["top_label"] = top_label
        row["top_label_score"] = top_score
        row["top_labels_json"] = _safe_json(top)
        row["timeline_event_count"] = sed_info.get("timeline_event_count")
        row["timeline_csv"] = sed_info.get("timeline_csv")
        row["timeline_jsonl"] = sed_info.get("timeline_jsonl")
        row["timeline_mode"] = sed_info.get("timeline_mode")
        row["timeline_inference_mode"] = sed_info.get("timeline_inference_mode")
        row["timeline_events_path"] = sed_info.get("timeline_events_path")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerow(row)


def write_moments_csv(
    path: Path,
    moments: Iterable[dict[str, Any]] | None,
    actions: Iterable[dict[str, Any]] | None,
    *,
    file_id: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "file_id",
        "kind",
        "timestamp",
        "speaker_id",
        "speaker_name",
        "arousal",
        "intent",
        "confidence",
        "text",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()

        def _write_rows(items: Iterable[dict[str, Any]], kind_override: str | None = None) -> None:
            for item in items:
                if not isinstance(item, dict):
                    continue
                row = {
                    "file_id": file_id or "",
                    "kind": kind_override or item.get("type") or "moment",
                    "timestamp": item.get("timestamp"),
                    "speaker_id": item.get("speaker_id") or item.get("speaker"),
                    "speaker_name": item.get("speaker_name"),
                    "arousal": item.get("arousal"),
                    "intent": item.get("intent"),
                    "confidence": item.get("confidence"),
                    "text": item.get("text") or item.get("description"),
                }
                writer.writerow(row)

        if moments:
            _write_rows(moments, kind_override="peak")
        if actions:
            _write_rows(actions, kind_override="action")


__all__ = [
    "SEGMENT_COLUMNS",
    "default_affect",
    "ensure_segment_keys",
    "write_segments_csv",
    "write_segments_jsonl",
    "write_timeline_csv",
    "write_narrative_report",
    "write_qc_report",
    "write_human_transcript",
    "write_speakers_summary",
    "write_conversation_metrics_csv",
    "write_overlap_summary_csv",
    "write_interruptions_csv",
    "write_audio_health_csv",
    "write_background_sed_summary_csv",
    "write_moments_csv",
]
