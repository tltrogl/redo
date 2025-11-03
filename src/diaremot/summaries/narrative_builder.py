"""Generate narrative summaries for human-friendly reports."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from statistics import mean
from typing import Any

from .conversation_analysis import ConversationMetrics


def _fmt_hms(seconds: float | None) -> str:
    try:
        value = max(0.0, float(seconds or 0.0))
    except (TypeError, ValueError):
        value = 0.0
    total_seconds = int(round(value))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _safe_mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return mean(vals) if vals else None


def _speaker_label(entry: dict[str, Any]) -> str:
    name = (entry or {}).get("speaker_name")
    if name and isinstance(name, str) and name.strip():
        return name.strip()
    sid = (entry or {}).get("speaker_id", "Speaker")
    return f"Speaker {sid}"


def build_narrative(
    segments: list[dict[str, Any]],
    speakers_summary: list[dict[str, Any]],
    conv_metrics: ConversationMetrics | None,
    *,
    total_duration: float | None = None,
) -> dict[str, Any]:
    """Compose a lightweight narrative summary from pipeline artefacts."""

    duration = total_duration
    if duration is None:
        if segments:
            try:
                duration = max(float(seg.get("end", 0) or 0.0) for seg in segments)
            except Exception:
                duration = None

    # Count unique speakers robustly (speakers_summary may contain duplicates).
    if speakers_summary:
        try:
            keys = []
            for s in speakers_summary:
                label = (s.get("speaker_name") or s.get("speaker_id") or "").strip()
                if label:
                    keys.append(label)
            num_speakers = len(set(keys)) if keys else len(speakers_summary)
        except Exception:
            num_speakers = len(speakers_summary)
    else:
        num_speakers = len({str(seg.get("speaker_id") or seg.get("speaker")) for seg in segments})

    # Dominant speaker
    dominant = None
    dominant_pct = None
    if conv_metrics and conv_metrics.speaker_dominance:
        try:
            speaker_id, share = max(
                conv_metrics.speaker_dominance.items(), key=lambda kv: float(kv[1] or 0.0)
            )
            dominant_pct = float(share or 0.0) * 100.0
            dominant_pct = max(0.0, min(100.0, dominant_pct))
            speaker_entry = next(
                (s for s in speakers_summary if str(s.get("speaker_id")) == str(speaker_id)),
                {"speaker_id": speaker_id, "speaker_name": speaker_id},
            )
            dominant = _speaker_label(speaker_entry)
        except Exception:
            dominant = None
            dominant_pct = None
    elif speakers_summary:
        # Aggregate durations by canonical label to avoid duplicate rows per speaker
        totals: dict[str, float] = {}
        for s in speakers_summary:
            label = _speaker_label(s)
            try:
                totals[label] = totals.get(label, 0.0) + float(s.get("total_duration", 0.0) or 0.0)
            except Exception:
                pass
        if totals:
            dominant, dom_sec = max(totals.items(), key=lambda kv: kv[1])
            total_sec = sum(totals.values()) or 1.0
            dominant_pct = 100.0 * dom_sec / total_sec
            dominant_pct = max(0.0, min(100.0, dominant_pct))

    # Emotion distribution
    emotion_counts = Counter(
        (seg.get("emotion_top") or "").strip().lower()
        for seg in segments
        if seg.get("emotion_top")
    )
    top_emotions = [
        (label.title(), count)
        for label, count in emotion_counts.most_common(3)
        if label and label != "unknown"
    ]

    avg_valence = _safe_mean(seg.get("valence") for seg in segments)
    avg_arousal = _safe_mean(seg.get("arousal") for seg in segments)
    avg_dominance = _safe_mean(seg.get("dominance") for seg in segments)

    # Build summary sentence
    duration_txt = _fmt_hms(duration)
    summary_parts: list[str] = [f"Conversation length {duration_txt}"]
    if num_speakers:
        summary_parts.append(f"{num_speakers} speaker{'s' if num_speakers != 1 else ''}")
    if dominant and dominant_pct:
        summary_parts.append(f"{dominant} led with approximately {dominant_pct:.0f}% share of airtime")
    summary = ", ".join(summary_parts) + "."

    if conv_metrics:
        pace = conv_metrics.conversation_pace_turns_per_min or 0.0
        summary += f" Pace averaged {pace:.1f} turns per minute."

    # Emotion brief
    if top_emotions:
        emotion_sentence = ", ".join(f"{label} ({count})" for label, count in top_emotions)
    else:
        emotion_sentence = "No dominant emotion detected; affect remained largely neutral."

    if avg_valence is not None:
        sentiment = "positive" if avg_valence > 0.15 else "negative" if avg_valence < -0.15 else "neutral"
        emotion_sentence += f" Overall valence trended {sentiment} (avg {avg_valence:.2f})."

    interaction_insights: list[str] = []
    if conv_metrics:
        balance = conv_metrics.turn_taking_balance or 0.0
        if balance >= 0.7:
            interaction_insights.append("Turn-taking was well balanced across speakers.")
        elif balance <= 0.4:
            interaction_insights.append("Conversation was strongly one-sided; one voice dominated.")
        else:
            interaction_insights.append("Participation was moderately balanced.")

        interrupt_rate = conv_metrics.interruption_rate_per_min or 0.0
        if interrupt_rate >= 3.0:
            interaction_insights.append(
                "Interruption rate was very high; dialogue may have been adversarial or manipulative."
            )
        elif interrupt_rate >= 1.0:
            interaction_insights.append("Interruption rate was elevated, suggesting competitive turn-taking.")
        else:
            interaction_insights.append("Interruption behaviour stayed low.")

        silence_ratio = conv_metrics.silence_ratio or 0.0
        if silence_ratio > 0.25:
            interaction_insights.append("Considerable portions of the session were silent.")
        elif silence_ratio < 0.05:
            interaction_insights.append("Conversation moved quickly with minimal pauses.")

        coherence = conv_metrics.topic_coherence_score or 0.0
        if coherence < 0.3:
            interaction_insights.append("Topic coherence was low; content shifted frequently.")
        elif coherence > 0.6:
            interaction_insights.append("Discussion stayed on topic with good coherence.")

        if dominant_pct and dominant_pct > 70.0:
            interaction_insights.append(
                "One participant held most of the floor, signalling a potential power imbalance."
            )

    if not interaction_insights:
        interaction_insights.append("Conversation metrics did not surface notable interaction risks.")

    emotion_stats = {
        "average_valence": avg_valence,
        "average_arousal": avg_arousal,
        "average_dominance": avg_dominance,
    }

    return {
        "summary": summary,
        "emotion_brief": emotion_sentence,
        "interaction_insights": interaction_insights,
        "top_emotions": top_emotions,
        "dominant_speaker": {"name": dominant, "percent": dominant_pct},
        "emotion_stats": emotion_stats,
    }


__all__ = ["build_narrative"]
