from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .config import DiarizedTurn


def collapse_single_speaker_turns(
    turns: Iterable[DiarizedTurn],
    *,
    dominance_threshold: float = 0.88,
    centroid_threshold: float = 0.08,
    min_turns: int = 3,
    secondary_max_ratio: float | None = None,
    secondary_min_duration_sec: float | None = None,
) -> tuple[bool, str | None, str | None]:
    turns = list(turns)
    if not turns or len(turns) <= 1:
        return False, None, None
    if min_turns > 1 and len(turns) < min_turns:
        return False, None, None
    speakers = {t.speaker for t in turns if t.speaker}
    if len(speakers) <= 1:
        return False, next(iter(speakers), None), None
    durations: dict[str, float] = {}
    total_duration = 0.0
    for t in turns:
        dur = max(float(t.end) - float(t.start), 0.0)
        durations[t.speaker] = durations.get(t.speaker, 0.0) + dur
        total_duration += dur
    if total_duration <= 0.0:
        return False, None, None
    dominant_speaker, dominant_duration = max(durations.items(), key=lambda item: item[1])
    dominance_ratio = dominant_duration / total_duration
    collapse_reason: str | None = None
    collapse = False
    if dominance_threshold > 0 and dominance_ratio >= dominance_threshold and len(durations) > 1:
        secondary_ratio = max(0.0, 1.0 - dominance_ratio)
        guard_limit = (
            secondary_max_ratio is not None
            and secondary_max_ratio > 0
            and secondary_ratio >= secondary_max_ratio
        )
        if guard_limit:
            return False, None, None
        collapse = True
        collapse_reason = f"dominance={dominance_ratio:.2f}"
    if not collapse and centroid_threshold > 0:
        by_speaker: dict[str, list[np.ndarray]] = {}
        for t in turns:
            if t.embedding is None:
                continue
            by_speaker.setdefault(t.speaker, []).append(np.asarray(t.embedding, dtype=np.float32))
        by_speaker = {k: v for k, v in by_speaker.items() if v}
        if len(by_speaker) >= 2:

            def centroid(vecs: list[np.ndarray]) -> np.ndarray:
                arr = np.vstack(vecs)
                c = arr.mean(axis=0)
                n = np.linalg.norm(c)
                return c / (n + 1e-9)

            centroids = {spk: centroid(vecs) for spk, vecs in by_speaker.items()}
            max_distance = 0.0
            keys = list(centroids.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    a = centroids[keys[i]]
                    b = centroids[keys[j]]
                    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
                    if denom <= 0:
                        continue
                    distance = 1.0 - float(np.dot(a, b) / denom)
                    if distance > max_distance:
                        max_distance = distance
            if max_distance <= centroid_threshold and len(keys) > 1:
                collapse = True
                collapse_reason = f"max_centroid_dist={max_distance:.3f}"
    if not collapse:
        return False, None, None
    if secondary_min_duration_sec is not None and secondary_min_duration_sec > 0:
        for speaker, duration in durations.items():
            if speaker == dominant_speaker:
                continue
            if duration >= secondary_min_duration_sec:
                return False, None, None
    canonical = dominant_speaker or "Speaker_1"
    if not canonical or canonical.lower() in {"", "none", "null"}:
        canonical = "Speaker_1"
    for turn in turns:
        turn.speaker = canonical
        turn.speaker_name = canonical
    return True, canonical, collapse_reason


__all__ = ["collapse_single_speaker_turns"]
