"""Aggregate analysis utilities for paralinguistic features."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .config import ParalinguisticsConfig


def analyze_speech_patterns_v2(features_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate statistics across multiple paralinguistic segments."""

    if not features_list:
        return {"error": "No features provided"}

    numeric_features = [
        "wpm",
        "sps",
        "disfluency_rate",
        "pause_ratio",
        "pitch_med_hz",
        "pitch_iqr_hz",
        "loudness_dbfs_med",
        "vq_jitter_pct",
        "vq_shimmer_db",
        "vq_hnr_db",
    ]

    analysis: dict[str, Any] = {}

    for feature_name in numeric_features:
        values = []
        for features in features_list:
            val = features.get(feature_name, np.nan)
            if not np.isnan(val) and val != 0.0:
                values.append(val)
        if values:
            analysis[f"{feature_name}_mean"] = float(np.mean(values))
            analysis[f"{feature_name}_std"] = float(np.std(values))
            analysis[f"{feature_name}_median"] = float(np.median(values))
            analysis[f"{feature_name}_min"] = float(np.min(values))
            analysis[f"{feature_name}_max"] = float(np.max(values))
        else:
            analysis[f"{feature_name}_mean"] = np.nan
            analysis[f"{feature_name}_std"] = np.nan
            analysis[f"{feature_name}_median"] = np.nan
            analysis[f"{feature_name}_min"] = np.nan
            analysis[f"{feature_name}_max"] = np.nan

    count_features = [
        "filler_count",
        "repetition_count",
        "false_start_count",
        "pause_count",
    ]

    for feature_name in count_features:
        values = [features.get(feature_name, 0) for features in features_list]
        analysis[f"{feature_name}_total"] = int(np.sum(values))
        analysis[f"{feature_name}_mean"] = float(np.mean(values))

    reliable_segments = sum(1 for f in features_list if f.get("vq_reliable", False))
    analysis["voice_quality_reliability"] = reliable_segments / len(features_list)

    analysis["total_segments"] = len(features_list)
    analysis["analysis_timestamp"] = time.time()

    return analysis


def detect_speech_anomalies_v2(features: dict[str, Any], reference_stats: dict[str, Any] | None = None) -> dict[str, Any]:
    """Detect speech anomalies for a single segment."""

    anomalies = {"flags": [], "severity_score": 0.0, "details": {}}

    wpm = features.get("wpm", np.nan)
    if not np.isnan(wpm):
        if wpm < 80:
            anomalies["flags"].append("very_slow_speech")
            anomalies["severity_score"] += 1.0
        elif wpm > 300:
            anomalies["flags"].append("very_fast_speech")
            anomalies["severity_score"] += 1.0
        anomalies["details"]["speech_rate"] = wpm

    disfluency_rate = features.get("disfluency_rate", 0.0)
    if disfluency_rate > 15.0:
        anomalies["flags"].append("high_disfluency")
        anomalies["severity_score"] += 1.5
    elif disfluency_rate > 8.0:
        anomalies["flags"].append("elevated_disfluency")
        anomalies["severity_score"] += 0.5
    anomalies["details"]["disfluency_rate"] = disfluency_rate

    pause_ratio = features.get("pause_ratio", 0.0)
    if pause_ratio > 0.5:
        anomalies["flags"].append("high_pause_ratio")
        anomalies["severity_score"] += 0.5
    anomalies["details"]["pause_ratio"] = pause_ratio

    loudness = features.get("loudness_dbfs_med", np.nan)
    if not np.isnan(loudness) and loudness < -40:
        anomalies["flags"].append("very_quiet")
        anomalies["severity_score"] += 0.5
    anomalies["details"]["loudness_dbfs_med"] = loudness

    jitter = features.get("vq_jitter_pct", 0.0)
    shimmer = features.get("vq_shimmer_db", 0.0)
    if jitter > 2.5:
        anomalies["flags"].append("high_jitter")
        anomalies["severity_score"] += 0.5
    if shimmer > 1.5:
        anomalies["flags"].append("high_shimmer")
        anomalies["severity_score"] += 0.5

    if reference_stats:
        ref_wpm = reference_stats.get("wpm_mean")
        if ref_wpm:
            delta = abs(wpm - ref_wpm)
            if delta > 60:
                anomalies["flags"].append("speech_rate_deviation")
                anomalies["severity_score"] += 0.5

    anomalies["details"]["severity_score"] = anomalies["severity_score"]
    return anomalies


def detect_backchannels_v2(
    segments: list[dict[str, Any]],
    cfg: ParalinguisticsConfig | None = None,
    *,
    window_sec: float = 1.2,
    min_pause_ratio: float = 0.35,
) -> dict[str, Any]:
    """Detect potential backchannel opportunities from pause patterns."""

    cfg = cfg or ParalinguisticsConfig()
    results = {
        "backchannel_candidates": [],
        "parameters": {
            "window_sec": window_sec,
            "min_pause_ratio": min_pause_ratio,
            "backchannel_max_ms": cfg.backchannel_max_ms,
        },
    }

    for idx, seg in enumerate(segments):
        pause_ratio = seg.get("pause_ratio", 0.0)
        duration = seg.get("duration_s", 0.0)
        if pause_ratio >= min_pause_ratio and duration <= (cfg.backchannel_max_ms / 1000.0):
            results["backchannel_candidates"].append({"segment_index": idx, "pause_ratio": pause_ratio, "duration_s": duration})

    return results


def compute_overlap_and_interruptions(
    segments: list[dict[str, Any]],
    min_overlap_sec: float = 0.05,
    interruption_gap_sec: float = 0.15,
) -> dict[str, Any]:
    """Compute overlap statistics using an event sweep over segment boundaries."""

    if not segments:
        return {
            "overlap_total_sec": 0.0,
            "overlap_ratio": 0.0,
            "by_speaker": {},
            "interruptions": [],
        }

    normalized: list[tuple[float, float, str, dict[str, Any]]] = []
    for seg in segments:
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", start) or start)
        if end < start:
            start, end = end, start
        speaker = str(seg.get("speaker_id") or seg.get("speaker") or "unknown")
        normalized.append((start, end, speaker, seg))

    if not normalized:
        return {
            "overlap_total_sec": 0.0,
            "overlap_ratio": 0.0,
            "by_speaker": {},
            "interruptions": [],
        }

    events: list[tuple[float, int, int]] = []
    for idx, (start, end, _, _) in enumerate(normalized):
        events.append((start, 1, idx))  # start events processed after overlap update
        events.append((end, 0, idx))  # ensure end events clear speakers before same-time starts

    events.sort(key=lambda event: (event[0], event[1]))

    total_start = min(event[0] for event in events if event[1] == 1)
    total_end = max(event[0] for event in events if event[1] == 0)
    total_dur = max(1e-6, total_end - total_start)

    overlap_total = 0.0
    by_speaker: dict[str, dict[str, Any]] = {}
    interruptions: list[dict[str, Any]] = []

    active_segments: dict[int, tuple[float, float, str, dict[str, Any]]] = {}

    for _, event_type, idx in events:
        if event_type == 0:  # end event
            active_segments.pop(idx, None)
            continue

        seg_start, seg_end, seg_speaker, _ = normalized[idx]

        for other_idx, other_seg in active_segments.items():
            other_start, other_end, other_speaker, _ = other_seg
            overlap_start = seg_start if seg_start >= other_start else other_start
            overlap_end = seg_end if seg_end <= other_end else other_end
            overlap = overlap_end - overlap_start
            if overlap < min_overlap_sec:
                continue

            overlap_total += overlap
            for speaker in (seg_speaker, other_speaker):
                slot = by_speaker.setdefault(
                    speaker, {"overlap_sec": 0.0, "interruptions": 0}
                )
                slot["overlap_sec"] = float(slot["overlap_sec"]) + overlap

            if seg_speaker != other_speaker:
                if seg_start > other_start:
                    later_speaker, later_time = seg_speaker, seg_start
                    earlier_speaker, earlier_time = other_speaker, other_start
                else:
                    later_speaker, later_time = other_speaker, other_start
                    earlier_speaker, earlier_time = seg_speaker, seg_start

                if 0.0 <= (later_time - earlier_time) <= interruption_gap_sec:
                    slot = by_speaker.setdefault(
                        later_speaker, {"overlap_sec": 0.0, "interruptions": 0}
                    )
                    slot["interruptions"] += 1
                    interruptions.append(
                        {
                            "at": float(later_time),
                            "interrupter": later_speaker,
                            "interrupted": earlier_speaker,
                            "overlap_sec": float(overlap),
                        }
                    )

        active_segments[idx] = normalized[idx]

    return {
        "overlap_total_sec": float(overlap_total),
        "overlap_ratio": float(overlap_total / total_dur) if total_dur > 0 else 0.0,
        "by_speaker": by_speaker,
        "interruptions": interruptions,
    }


__all__ = [
    "analyze_speech_patterns_v2",
    "detect_speech_anomalies_v2",
    "detect_backchannels_v2",
    "compute_overlap_and_interruptions",
]
