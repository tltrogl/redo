"""
speakers_summary_builder.py — Enriched rollups from diarized_transcript_with_emotion.csv

Input: segments list with per-segment data
Output: List of speaker summary dicts

Notes:
- Interruptions/overlap are provided via per_speaker_interrupts parameter
- We aggregate text emotions using the FULL 28-class distribution when present (preferred)
- avg_wpm = (total_words / total_duration_minutes). WPM from segments (if present) is used only as a backup
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

GOEMOTIONS_LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]


def _float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_json(s: Any) -> Any:
    if isinstance(s, (dict, list)):
        return s
    if isinstance(s, str) and s.strip():
        try:
            return json.loads(s)
        except Exception:
            return None
    return None


def _words(text: str) -> int:
    if not text:
        return 0
    return sum(1 for tok in str(text).strip().split() if tok)


def _top_k(d: dict[str, float], k: int = 5) -> list[tuple[str, float]]:
    return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:k]


def _normalize(d: dict[str, float]) -> dict[str, float]:
    total = sum(d.values()) or 1.0
    return {k: (v / total) for k, v in d.items()}


@dataclass
class RunningMoments:
    """Track sample statistics using Welford's algorithm."""

    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    min_value: float | None = None
    max_value: float | None = None

    def add(self, value: float | None) -> None:
        if value is None:
            return
        try:
            val = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(val):
            return
        self.count += 1
        self.total += val
        self.total_sq += val * val
        if self.min_value is None or val < self.min_value:
            self.min_value = val
        if self.max_value is None or val > self.max_value:
            self.max_value = val

    def mean(self, default: float = 0.0) -> float:
        if self.count == 0:
            return default
        return self.total / self.count

    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        mean = self.total / self.count
        return max(0.0, (self.total_sq / self.count) - (mean * mean))

    def stddev(self) -> float:
        return math.sqrt(self.variance())


@dataclass
class P2QuantileEstimator:
    """Streaming quantile estimator (P² algorithm)."""

    probability: float
    _initial: list[float] = field(default_factory=list)
    _q: list[float] = field(default_factory=list)
    _n: list[int] = field(default_factory=list)
    _np: list[float] = field(default_factory=list)
    _dn: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not (0.0 < self.probability < 1.0):
            raise ValueError("probability must be between 0 and 1")

    def add(self, value: float | None) -> None:
        if value is None:
            return
        try:
            sample = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(sample):
            return

        if len(self._q) < 5:
            self._initial.append(sample)
            if len(self._initial) == 5:
                self._initial.sort()
                self._q = self._initial[:]
                self._n = [i for i in range(1, 6)]
                p = self.probability
                self._np = [1.0, 1.0 + 2.0 * p, 1.0 + 4.0 * p, 3.0 + 2.0 * p, 5.0]
                self._dn = [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0]
                self._initial.clear()
            return

        q = self._q
        if sample < q[0]:
            q[0] = sample
            k = 0
        elif sample >= q[4]:
            q[4] = sample
            k = 3
        else:
            k = 0
            while k < 4 and sample >= q[k + 1]:
                k += 1

        for i in range(k + 1, 5):
            self._n[i] += 1

        for i in range(5):
            self._np[i] += self._dn[i]

        for i in range(1, 4):
            diff = self._np[i] - self._n[i]
            if (diff >= 1.0 and self._n[i + 1] - self._n[i] > 1) or (
                diff <= -1.0 and self._n[i - 1] - self._n[i] < -1
            ):
                direction = 1 if diff > 0 else -1
                candidate = self._parabolic(i, direction)
                if q[i - 1] < candidate < q[i + 1]:
                    q[i] = candidate
                else:
                    q[i] = self._linear(i, direction)
                self._n[i] += direction

    def _parabolic(self, idx: int, direction: int) -> float:
        q = self._q
        n = self._n
        return q[idx] + (direction / (n[idx + 1] - n[idx - 1])) * (
            (n[idx] - n[idx - 1] + direction) * (q[idx + 1] - q[idx]) / (n[idx + 1] - n[idx])
            + (n[idx + 1] - n[idx] - direction) * (q[idx] - q[idx - 1]) / (n[idx] - n[idx - 1])
        )

    def _linear(self, idx: int, direction: int) -> float:
        q = self._q
        n = self._n
        return q[idx] + direction * (q[idx + direction] - q[idx]) / (n[idx + direction] - n[idx])

    def result(self, default: float = 0.0) -> float:
        if self._q:
            return float(self._q[2])
        if not self._initial:
            return default
        data = sorted(self._initial)
        index = int(round(self.probability * (len(data) - 1)))
        index = max(0, min(len(data) - 1, index))
        return float(data[index])


@dataclass
class SpeakerAccumulator:
    speaker_id: str
    speaker_name: str
    duration_total: float = 0.0
    words_total: int = 0
    per_seg_wpm: P2QuantileEstimator = field(default_factory=lambda: P2QuantileEstimator(0.5))
    valence: P2QuantileEstimator = field(default_factory=lambda: P2QuantileEstimator(0.5))
    arousal: P2QuantileEstimator = field(default_factory=lambda: P2QuantileEstimator(0.5))
    dominance: P2QuantileEstimator = field(default_factory=lambda: P2QuantileEstimator(0.5))
    ser_votes: dict[str, int] = field(default_factory=dict)
    text_full_sum: dict[str, float] = field(
        default_factory=lambda: {label: 0.0 for label in GOEMOTIONS_LABELS}
    )
    text_top_sum: dict[str, float] = field(default_factory=dict)
    f0_mean: P2QuantileEstimator = field(default_factory=lambda: P2QuantileEstimator(0.5))
    f0_std: P2QuantileEstimator = field(default_factory=lambda: P2QuantileEstimator(0.5))
    loudness_q1: P2QuantileEstimator = field(default_factory=lambda: P2QuantileEstimator(0.25))
    loudness_q2: P2QuantileEstimator = field(default_factory=lambda: P2QuantileEstimator(0.5))
    loudness_q3: P2QuantileEstimator = field(default_factory=lambda: P2QuantileEstimator(0.75))
    loudness_stats: RunningMoments = field(default_factory=RunningMoments)
    pause_total_sec: float = 0.0
    pause_count: int = 0
    interruptions_made: int = 0
    interruptions_received: int = 0
    overlap_ratio: float | None = None
    vq_jitter: P2QuantileEstimator = field(default_factory=lambda: P2QuantileEstimator(0.5))
    vq_shimmer: P2QuantileEstimator = field(default_factory=lambda: P2QuantileEstimator(0.5))
    vq_hnr: P2QuantileEstimator = field(default_factory=lambda: P2QuantileEstimator(0.5))
    vq_cpps: P2QuantileEstimator = field(default_factory=lambda: P2QuantileEstimator(0.5))
    vq_hint_votes: dict[str, int] = field(default_factory=dict)

    def update(self, row: dict[str, Any]) -> None:
        start = _float(row.get("start"))
        end = _float(row.get("end"))
        duration = max(0.0, end - start)
        self.duration_total += duration

        words = row.get("words")
        if words in (None, ""):
            words = _words(row.get("text", ""))
        try:
            self.words_total += int(words)
        except (TypeError, ValueError):
            self.words_total += _words(row.get("text", ""))

        if row.get("wpm") not in (None, ""):
            self.per_seg_wpm.add(row.get("wpm"))

        for key, estimator in (
            ("valence", self.valence),
            ("arousal", self.arousal),
            ("dominance", self.dominance),
        ):
            if row.get(key) not in (None, ""):
                estimator.add(row.get(key))

        emo_top = row.get("emotion_top") or ""
        if emo_top:
            self.ser_votes[emo_top] = self.ser_votes.get(emo_top, 0) + 1

        payload = _extract_affect_payload(row)
        full_payload = payload.get("text_full")
        if isinstance(full_payload, dict):
            for label, score in full_payload.items():
                if label in self.text_full_sum:
                    self.text_full_sum[label] += _float(score)
        else:
            top_payload = payload.get("text_top")
            if isinstance(top_payload, list):
                for item in top_payload:
                    if isinstance(item, dict) and "label" in item and "score" in item:
                        label = str(item["label"])
                        self.text_top_sum[label] = self.text_top_sum.get(label, 0.0) + _float(
                            item["score"]
                        )
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        label = str(item[0])
                        self.text_top_sum[label] = self.text_top_sum.get(label, 0.0) + _float(
                            item[1]
                        )

        self.f0_mean.add(row.get("f0_mean_hz"))
        self.f0_std.add(row.get("f0_std_hz"))
        self.loudness_q1.add(row.get("loudness_rms"))
        self.loudness_q2.add(row.get("loudness_rms"))
        self.loudness_q3.add(row.get("loudness_rms"))
        self.loudness_stats.add(row.get("loudness_rms"))

        if row.get("pause_time_s") not in (None, ""):
            self.pause_total_sec += _float(row.get("pause_time_s"))
        if row.get("pause_count") not in (None, ""):
            try:
                self.pause_count += int(float(row.get("pause_count")))
            except (TypeError, ValueError):
                pass

        if row.get("interruptions_made") not in (None, ""):
            try:
                self.interruptions_made += int(float(row.get("interruptions_made")))
            except (TypeError, ValueError):
                pass
        if row.get("interruptions_received") not in (None, ""):
            try:
                self.interruptions_received += int(float(row.get("interruptions_received")))
            except (TypeError, ValueError):
                pass

        if row.get("overlap_ratio") not in (None, ""):
            value = _float(row.get("overlap_ratio"))
            self.overlap_ratio = value if self.overlap_ratio is None else max(self.overlap_ratio, value)

        self.vq_jitter.add(row.get("vq_jitter_pct"))
        self.vq_shimmer.add(row.get("vq_shimmer_db"))
        self.vq_hnr.add(row.get("vq_hnr_db"))
        self.vq_cpps.add(row.get("vq_cpps_db"))

        hint = row.get("voice_quality_hint")
        if hint not in (None, ""):
            label = str(hint).strip()
            if label:
                self.vq_hint_votes[label] = self.vq_hint_votes.get(label, 0) + 1

    def merge_interruptions(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        if payload.get("made") not in (None, ""):
            try:
                self.interruptions_made = int(float(payload.get("made")))
            except (TypeError, ValueError):
                pass
        if payload.get("received") not in (None, ""):
            try:
                self.interruptions_received = int(float(payload.get("received")))
            except (TypeError, ValueError):
                pass
        overlap_sec = payload.get("overlap_sec")
        if overlap_sec not in (None, "") and self.duration_total > 0:
            try:
                overlap_val = float(overlap_sec)
            except (TypeError, ValueError):
                overlap_val = 0.0
            self.overlap_ratio = max(0.0, overlap_val) / max(self.duration_total, 1e-9)

    def _median(self, estimator: P2QuantileEstimator, default: float = 0.0) -> float:
        return float(estimator.result(default))

    def _text_distribution(self) -> tuple[dict[str, float], list[dict[str, float]]]:
        if any(value > 0.0 for value in self.text_full_sum.values()):
            normalized = _normalize(self.text_full_sum)
        elif self.text_top_sum:
            filtered = {
                key: value for key, value in self.text_top_sum.items() if key in GOEMOTIONS_LABELS
            } or {"neutral": 1.0}
            normalized = _normalize(filtered)
        else:
            normalized = {"neutral": 1.0}
        top5 = [{"label": key, "score": float(score)} for key, score in _top_k(normalized, 5)]
        return normalized, top5

    def finalize(self) -> dict[str, Any]:
        total_dur = self.duration_total
        total_minutes = total_dur / 60.0 if total_dur > 0 else 0.0
        avg_wpm = (self.words_total / total_minutes) if total_minutes > 0 else self._median(self.per_seg_wpm, 0.0)

        text_full, text_top5 = self._text_distribution()
        dominant_text = text_top5[0]["label"] if text_top5 else "neutral"
        dominant_speech = max(self.ser_votes, key=self.ser_votes.get) if self.ser_votes else "neutral"

        loud_q1 = self.loudness_q1.result(self.loudness_stats.min_value or 0.0)
        loud_q3 = self.loudness_q3.result(self.loudness_stats.max_value or 0.0)
        loud_med = self.loudness_q2.result(0.0)
        loud_dr = max(0.0, (float(loud_q3) - float(loud_q1)) * 20.0) if self.loudness_stats.count else 0.0

        pause_ratio = (self.pause_total_sec / total_dur) if total_dur > 0 else 0.0
        voice_hint = max(self.vq_hint_votes, key=self.vq_hint_votes.get) if self.vq_hint_votes else ""

        return {
            "speaker_id": self.speaker_id,
            "speaker_name": self.speaker_name,
            "total_duration": round(total_dur, 3),
            "word_count": int(self.words_total),
            "avg_wpm": round(avg_wpm, 1),
            "avg_valence": round(self._median(self.valence, 0.0), 3),
            "avg_arousal": round(self._median(self.arousal, 0.0), 3),
            "avg_dominance": round(self._median(self.dominance, 0.0), 3),
            "dominant_speech_emotion": dominant_speech,
            "dominant_text_emotion": dominant_text,
            "interruptions_made": int(self.interruptions_made),
            "interruptions_received": int(self.interruptions_received),
            "overlap_ratio": round(self.overlap_ratio, 3) if self.overlap_ratio is not None else "",
            "f0_mean_hz": round(self._median(self.f0_mean, 0.0), 2),
            "f0_std_hz": round(self._median(self.f0_std, 0.0), 2),
            "loudness_rms_med": round(float(loud_med), 3),
            "loudness_dr_db": round(float(loud_dr), 2),
            "pause_total_sec": round(self.pause_total_sec, 2),
            "pause_count": int(self.pause_count),
            "pause_ratio": round(pause_ratio, 3),
            "vq_jitter_pct_med": round(self._median(self.vq_jitter, 0.0), 3),
            "vq_shimmer_db_med": round(self._median(self.vq_shimmer, 0.0), 3),
            "vq_hnr_db_med": round(self._median(self.vq_hnr, 0.0), 3),
            "vq_cpps_db_med": round(self._median(self.vq_cpps, 0.0), 3),
            "voice_quality_hint": voice_hint,
            "text_emotions_top5_json": json.dumps(text_top5, ensure_ascii=False),
            "text_emotions_full_json": json.dumps(text_full, ensure_ascii=False),
        }


def _extract_affect_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("_affect_payload")
    if isinstance(payload, dict):
        return payload

    full = _safe_json(row.get("text_emotions_full_json"))
    top = _safe_json(row.get("text_emotions_top5_json"))
    payload = {
        "text_full": full if isinstance(full, dict) else None,
        "text_top": top if isinstance(top, list) else None,
    }
    row["_affect_payload"] = payload
    return payload


def build_speakers_summary(
    segments: list[dict[str, Any]] | None,
    per_speaker_interrupts: dict[str, dict[str, Any]] | None,
    overlap_stats: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Build a speaker summary compatible with existing CSV outputs."""

    accumulators: dict[str, SpeakerAccumulator] = {}

    for row in segments or []:
        if not isinstance(row, dict):
            continue
        speaker_id = row.get("speaker_id") or "Unknown"
        speaker_name = row.get("speaker_name") or ""
        acc = accumulators.get(speaker_id)
        if acc is None:
            acc = SpeakerAccumulator(speaker_id=speaker_id, speaker_name=speaker_name)
            accumulators[speaker_id] = acc
        elif not acc.speaker_name and speaker_name:
            acc.speaker_name = speaker_name
        acc.update(row)

    for speaker_id, payload in (per_speaker_interrupts or {}).items():
        if speaker_id in accumulators:
            accumulators[speaker_id].merge_interruptions(payload)

    return [acc.finalize() for acc in accumulators.values()]
