"""Conversation flow analysis and metrics calculation with robust error handling."""

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

try:  # pragma: no cover - import guard behaviour is exercised via tests
    import pandas as pd
except Exception:  # noqa: S110 - broad except to gracefully fall back when pandas is unavailable
    pd = None  # type: ignore[assignment]


@dataclass
class ConversationMetrics:
    turn_taking_balance: float  # entropy of speaker participation
    interruption_rate_per_min: float
    avg_turn_duration_sec: float
    conversation_pace_turns_per_min: float
    silence_ratio: float
    speaker_dominance: dict[str, float]  # speaker_id -> % of total time
    response_latency_stats: dict[str, float]  # avg, median response times
    topic_coherence_score: float  # 0-1, semantic consistency
    energy_flow: list[dict[str, Any]]  # time-series of engagement
    interruptions_per_speaker: dict[str, float] = field(default_factory=dict)


def analyze_conversation_flow(
    segments: list[dict[str, Any]], total_duration_sec: float
) -> ConversationMetrics:
    """Analyze conversation patterns and dynamics with robust error handling."""
    try:
        if not segments or total_duration_sec <= 0:
            return _empty_metrics()

        # Sort segments by time safely
        try:
            segs = sorted(segments, key=lambda x: float(x.get("start", 0) or 0))
        except (TypeError, ValueError):
            segs = segments  # Use original order if sorting fails

        sanitized = list(_sanitize_segments(segs))
        if not sanitized:
            return _empty_metrics()

        analysis = None
        if pd is not None:
            analysis = _analyze_conversation_flow_vectorized(sanitized, total_duration_sec)

        if analysis is None:
            analysis = _analyze_conversation_flow_fallback(sanitized, total_duration_sec)

        if analysis is None:
            return _empty_metrics()

        balance = analysis["turn_taking_balance"]
        dominance = analysis["speaker_dominance"]
        avg_turn_duration = analysis["avg_turn_duration_sec"]
        turns_per_min = analysis["conversation_pace_turns_per_min"]
        interrupt_rate = analysis["interruption_rate_per_min"]
        interrupts_per_speaker = analysis["interruptions_per_speaker"]
        response_stats = analysis["response_latency_stats"]
        total_speech_time = analysis["total_speech_time"]

        # 4. Topic coherence with error handling
        coherence = _calculate_topic_coherence(segs)

        # 5. Energy flow over time with bounds checking
        energy_flow = _calculate_energy_flow(segs, total_duration_sec)

        # 6. Silence ratio with safe division
        silence_ratio = 0.0
        try:
            silence_ratio = max(
                0.0,
                min(1.0, (total_duration_sec - total_speech_time) / total_duration_sec),
            )
        except ZeroDivisionError:
            silence_ratio = 1.0

        return ConversationMetrics(
            turn_taking_balance=float(balance),
            interruption_rate_per_min=float(interrupt_rate),
            avg_turn_duration_sec=float(avg_turn_duration),
            conversation_pace_turns_per_min=float(turns_per_min),
            silence_ratio=float(silence_ratio),
            speaker_dominance=dominance,
            response_latency_stats=response_stats,
            topic_coherence_score=float(coherence),
            energy_flow=energy_flow,
            interruptions_per_speaker=interrupts_per_speaker,
        )

    except Exception as e:
        # Fallback to empty metrics on any error
        print(f"Warning: conversation analysis failed: {e}")
        return _empty_metrics()


def _sanitize_segments(segs: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    """Yield sanitized segment records with numeric bounds."""

    for seg in segs:
        try:
            speaker = seg.get("speaker_id") or seg.get("speaker") or "Unknown"
            start = float(seg.get("start", 0) or 0)
            end = float(seg.get("end", 0) or 0)
            duration = max(0.0, end - start)
            arousal_val = seg.get("arousal", 0) or 0.0
            try:
                arousal = float(arousal_val)
            except (TypeError, ValueError):
                arousal = 0.0
            yield {
                "speaker": str(speaker),
                "start": start,
                "end": end,
                "duration": duration,
                "arousal": arousal,
                "text": seg.get("text", ""),
            }
        except (TypeError, ValueError):
            continue


def _analyze_conversation_flow_vectorized(
    sanitized: list[dict[str, Any]], total_duration_sec: float
) -> dict[str, Any] | None:
    """Vectorised metric computation using pandas."""

    if pd is None or not sanitized:
        return None

    df = pd.DataFrame.from_records(sanitized)
    if df.empty:
        return None

    durations = df["duration"].astype(float)
    total_speech_time = float(durations.sum())
    if total_speech_time <= 0:
        return None

    speaker_totals = df.groupby("speaker")["duration"].sum()
    dominance = (
        speaker_totals.div(total_speech_time)
        .mul(100.0)
        .astype(float)
        .to_dict()
    )

    balance = 0.0
    if len(speaker_totals) > 1:
        probs = speaker_totals.div(total_speech_time)
        probs = probs[probs > 0]
        if not probs.empty:
            entropy = float(-(probs * np.log2(probs)).sum())
            max_entropy = np.log2(len(probs))
            balance = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    df = df.assign(duration=durations)
    prev_end = df["end"].shift(fill_value=0.0)
    prev_speaker = df["speaker"].shift()
    overlap_mask = prev_speaker.notna() & (df["start"] < prev_end)
    interrupt_counts = (
        df.loc[overlap_mask, "speaker"].value_counts().astype(float).to_dict()
    )

    duration_minutes = total_duration_sec / 60.0 if total_duration_sec > 0 else 0.0
    if duration_minutes > 0:
        interrupt_rate = sum(interrupt_counts.values()) / duration_minutes
        interrupts_per_speaker = {
            spk: count / duration_minutes for spk, count in interrupt_counts.items()
        }
    else:
        interrupt_rate = 0.0
        interrupts_per_speaker = {
            spk: 0.0 for spk in interrupt_counts.keys()
        }

    turn_mask = (
        (df.index == df.index[0]) | (df["speaker"] != df["speaker"].shift())
    ) & (df["duration"] > 0)
    turns_df = df.loc[turn_mask, ["speaker", "start", "end", "duration"]]

    avg_turn_duration = float(turns_df["duration"].mean()) if not turns_df.empty else 0.0
    turn_count = int(len(turns_df))
    if duration_minutes > 0:
        turns_per_min = turn_count / duration_minutes
    else:
        turns_per_min = 0.0

    latencies = turns_df["start"] - turns_df["end"].shift()
    latencies = latencies[latencies.notna()]
    latency_window = latencies[(latencies >= 0.0) & (latencies <= 5.0)]
    response_stats = {
        "avg_sec": float(latency_window.mean()) if not latency_window.empty else 0.0,
        "median_sec": float(latency_window.median()) if not latency_window.empty else 0.0,
        "count": int(len(latency_window)),
    }

    return {
        "turn_taking_balance": float(balance),
        "speaker_dominance": {str(k): float(v) for k, v in dominance.items()},
        "avg_turn_duration_sec": float(avg_turn_duration),
        "conversation_pace_turns_per_min": float(turns_per_min),
        "interruption_rate_per_min": float(interrupt_rate),
        "interruptions_per_speaker": {
            str(k): float(v) for k, v in interrupts_per_speaker.items()
        },
        "response_latency_stats": response_stats,
        "total_speech_time": float(total_speech_time),
    }


def _analyze_conversation_flow_fallback(
    sanitized: list[dict[str, Any]], total_duration_sec: float
) -> dict[str, Any] | None:
    """Fallback implementation mirroring the legacy per-segment logic."""

    speaker_times: dict[str, float] = {}
    total_speech_time = 0.0
    for seg in sanitized:
        speaker = seg["speaker"]
        duration = seg["duration"]
        speaker_times[speaker] = speaker_times.get(speaker, 0.0) + duration
        total_speech_time += duration

    if not speaker_times or total_speech_time <= 0:
        return None

    dominance: dict[str, float] = {}
    for spk, time in speaker_times.items():
        try:
            dominance[spk] = (time / total_speech_time) * 100.0
        except ZeroDivisionError:
            dominance[spk] = 0.0

    balance = 0.0
    if len(speaker_times) > 1 and total_speech_time > 0:
        try:
            probs = [t / total_speech_time for t in speaker_times.values()]
            probs = [p for p in probs if p > 0]
            if probs:
                entropy = -sum(p * np.log2(p) for p in probs)
                max_entropy = np.log2(len(probs))
                balance = entropy / max_entropy if max_entropy > 0 else 0.0
        except (ValueError, ZeroDivisionError):
            balance = 0.0

    turns: list[dict[str, float]] = []
    prev_speaker: str | None = None
    prev_end = 0.0
    interrupts: dict[str, int] = {}
    for seg in sanitized:
        current_speaker = seg["speaker"]
        start = seg["start"]
        end = seg["end"]
        duration = seg["duration"]

        if current_speaker != prev_speaker and duration > 0:
            turns.append(
                {
                    "speaker": current_speaker,
                    "start": start,
                    "end": end,
                    "duration": duration,
                }
            )
        if prev_speaker is not None and start < prev_end:
            interrupts[current_speaker] = interrupts.get(current_speaker, 0) + 1
        prev_speaker = current_speaker
        prev_end = end

    turn_durations = [t["duration"] for t in turns if t.get("duration", 0) > 0]
    avg_turn_duration = float(np.mean(turn_durations)) if turn_durations else 0.0

    try:
        duration_minutes = total_duration_sec / 60.0
        turns_per_min = len(turns) / duration_minutes if duration_minutes > 0 else 0.0
    except ZeroDivisionError:
        turns_per_min = 0.0
        duration_minutes = 0.0

    try:
        interrupt_rate = (
            sum(interrupts.values()) / duration_minutes if duration_minutes > 0 else 0.0
        )
        interrupts_per_speaker = {
            spk: (cnt / duration_minutes if duration_minutes > 0 else 0.0)
            for spk, cnt in interrupts.items()
        }
    except Exception:
        interrupt_rate = 0.0
        interrupts_per_speaker = {}

    response_times = []
    for i in range(1, len(turns)):
        try:
            prev_end_val = turns[i - 1]["end"]
            curr_start = turns[i]["start"]
            latency = curr_start - prev_end_val
            if 0 <= latency <= 5.0:
                response_times.append(latency)
        except (KeyError, TypeError, ValueError):
            continue

    response_stats = {
        "avg_sec": float(np.mean(response_times)) if response_times else 0.0,
        "median_sec": float(np.median(response_times)) if response_times else 0.0,
        "count": len(response_times),
    }

    return {
        "turn_taking_balance": float(balance),
        "speaker_dominance": dominance,
        "avg_turn_duration_sec": float(avg_turn_duration),
        "conversation_pace_turns_per_min": float(turns_per_min),
        "interruption_rate_per_min": float(interrupt_rate),
        "interruptions_per_speaker": interrupts_per_speaker,
        "response_latency_stats": response_stats,
        "total_speech_time": float(total_speech_time),
    }


def _calculate_topic_coherence(segments: list[dict[str, Any]]) -> float:
    """Calculate topic coherence with error handling."""
    try:
        if len(segments) < 2:
            return 1.0

        coherence_scores = []

        for i in range(1, len(segments)):
            try:
                prev_text = str(segments[i - 1].get("text", "")).lower().split()
                curr_text = str(segments[i].get("text", "")).lower().split()

                if not prev_text or not curr_text:
                    continue

                # Simple keyword overlap - only content words
                prev_words = set(w for w in prev_text if len(w) > 3)
                curr_words = set(w for w in curr_text if len(w) > 3)

                if prev_words and curr_words:
                    overlap = len(prev_words & curr_words)
                    union = len(prev_words | curr_words)
                    if union > 0:
                        jaccard = overlap / union
                        coherence_scores.append(jaccard)
            except (TypeError, AttributeError):
                continue

        return np.mean(coherence_scores) if coherence_scores else 0.0

    except Exception:
        return 0.0


def _calculate_energy_flow(
    segments: list[dict[str, Any]], total_duration: float
) -> list[dict[str, Any]]:
    """Calculate engagement/energy over time windows with error handling."""
    try:
        if total_duration <= 0:
            return []

        window_sec = 30.0  # 30-second windows
        num_windows = max(1, int(total_duration / window_sec))

        flow = []
        for i in range(num_windows):
            try:
                window_start = i * window_sec
                window_end = min((i + 1) * window_sec, total_duration)

                # Find segments in this window with bounds checking
                window_segs = []
                for s in segments:
                    try:
                        seg_start = float(s.get("start", 0) or 0)
                        seg_end = float(s.get("end", 0) or 0)
                        if seg_start < window_end and seg_end > window_start:
                            window_segs.append(s)
                    except (TypeError, ValueError):
                        continue

                if not window_segs:
                    energy = 0.0
                    speakers = 0
                    speech_density = 0.0
                else:
                    # Energy = average arousal * speech density
                    arousals = []
                    for s in window_segs:
                        try:
                            arousal = float(s.get("arousal", 0) or 0)
                            arousals.append(arousal)
                        except (TypeError, ValueError):
                            arousals.append(0.0)

                    avg_arousal = np.mean(arousals) if arousals else 0.0

                    # Calculate speech time in window
                    speech_time = 0.0
                    for s in window_segs:
                        try:
                            seg_start = float(s.get("start", 0) or 0)
                            seg_end = float(s.get("end", 0) or 0)
                            # Clip to window boundaries
                            clipped_start = max(seg_start, window_start)
                            clipped_end = min(seg_end, window_end)
                            if clipped_end > clipped_start:
                                speech_time += clipped_end - clipped_start
                        except (TypeError, ValueError):
                            continue

                    speech_density = speech_time / window_sec if window_sec > 0 else 0.0
                    energy = avg_arousal * speech_density

                    # Count unique speakers
                    speakers = len(
                        set(
                            s.get("speaker_id") or s.get("speaker") or "Unknown"
                            for s in window_segs
                        )
                    )

                flow.append(
                    {
                        "window_start_sec": window_start,
                        "window_end_sec": window_end,
                        "energy_score": float(energy),
                        "active_speakers": int(speakers),
                        "speech_density": float(speech_density),
                    }
                )

            except Exception:
                # Skip this window on error
                continue

        return flow

    except Exception:
        return []


def _empty_metrics() -> ConversationMetrics:
    """Return empty/default metrics for edge cases."""
    return ConversationMetrics(
        turn_taking_balance=0.0,
        interruption_rate_per_min=0.0,
        avg_turn_duration_sec=0.0,
        conversation_pace_turns_per_min=0.0,
        silence_ratio=1.0,
        speaker_dominance={},
        response_latency_stats={"avg_sec": 0.0, "median_sec": 0.0, "count": 0},
        topic_coherence_score=0.0,
        energy_flow=[],
        interruptions_per_speaker={},
    )


def build_conversation_analysis(
    segments: list[dict[str, Any]],
    total_duration_sec: float,
    overlap_stats: dict[str, Any],
) -> dict[str, Any]:
    """Core interface: build conversation analysis from pipeline data"""
    metrics = analyze_conversation_flow(segments, total_duration_sec)

    return {
        "turn_taking_balance": metrics.turn_taking_balance,
        "interruption_rate_per_min": metrics.interruption_rate_per_min,
        "avg_turn_duration_sec": metrics.avg_turn_duration_sec,
        "conversation_pace_turns_per_min": metrics.conversation_pace_turns_per_min,
        "silence_ratio": metrics.silence_ratio,
        "speaker_dominance": metrics.speaker_dominance,
        "response_latency_stats": metrics.response_latency_stats,
        "topic_coherence_score": metrics.topic_coherence_score,
        "energy_flow": metrics.energy_flow,
        "interruptions_per_speaker": metrics.interruptions_per_speaker,
        "overlap_stats": overlap_stats,
    }
