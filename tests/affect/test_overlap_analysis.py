from __future__ import annotations

import math
import time
from copy import deepcopy
from statistics import median

from diaremot.affect.paralinguistics.analysis import compute_overlap_and_interruptions


def _legacy_overlap(
    segments: list[dict[str, float]],
    *,
    min_overlap_sec: float = 0.05,
    interruption_gap_sec: float = 0.15,
) -> dict[str, object]:
    """Reference implementation matching the pre-sweep-line behaviour."""

    if not segments:
        return {
            "overlap_total_sec": 0.0,
            "overlap_ratio": 0.0,
            "by_speaker": {},
            "interruptions": [],
        }

    norm: list[tuple[float, float, str, dict[str, float]]] = []
    for seg in segments:
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", start) or start)
        if end < start:
            start, end = end, start
        speaker = str(seg.get("speaker_id") or seg.get("speaker") or "unknown")
        norm.append((start, end, speaker, seg))

    norm.sort(key=lambda item: item[0])

    total_start = norm[0][0]
    total_end = max(entry[1] for entry in norm)
    total_dur = max(1e-6, total_end - total_start)

    overlap_total = 0.0
    by_speaker: dict[str, dict[str, object]] = {}
    interruptions: list[dict[str, object]] = []

    j = 0
    for i, (si, ei, spk_i, _) in enumerate(norm):
        while j < i and norm[j][1] <= si:
            j += 1
        for k in range(j, i):
            sk, ek, spk_k, _ = norm[k]
            start = max(si, sk)
            end = min(ei, ek)
            overlap = end - start
            if overlap >= min_overlap_sec:
                overlap_total += overlap
                for speaker in (spk_i, spk_k):
                    slot = by_speaker.setdefault(
                        speaker, {"overlap_sec": 0.0, "interruptions": 0}
                    )
                    slot["overlap_sec"] = float(slot["overlap_sec"]) + overlap

                if spk_i != spk_k:
                    later = (spk_i, si) if si > sk else (spk_k, sk)
                    earlier = (spk_k, sk) if si > sk else (spk_i, si)
                    if 0.0 <= (later[1] - earlier[1]) <= interruption_gap_sec:
                        by_speaker.setdefault(
                            later[0], {"overlap_sec": 0.0, "interruptions": 0}
                        )["interruptions"] += 1
                        interruptions.append(
                            {
                                "at": float(later[1]),
                                "interrupter": later[0],
                                "interrupted": earlier[0],
                                "overlap_sec": float(overlap),
                            }
                        )

    return {
        "overlap_total_sec": float(overlap_total),
        "overlap_ratio": float(overlap_total / total_dur) if total_dur > 0 else 0.0,
        "by_speaker": by_speaker,
        "interruptions": interruptions,
    }


def _dense_turn_segments(
    speakers: int = 3,
    turns_per_speaker: int = 25,
    *,
    base_gap: float = 0.12,
    duration: float = 0.45,
) -> list[dict[str, float]]:
    segments: list[dict[str, float]] = []
    for turn in range(speakers * turns_per_speaker):
        speaker = f"S{turn % speakers}"
        start = turn * base_gap
        jitter = (turn % 5) * 0.01
        segments.append({
            "start": start,
            "end": start + duration + jitter,
            "speaker": speaker,
        })
    return segments


def _sort_interruptions(data: list[dict[str, object]]) -> list[tuple[str, str, float, float]]:
    return sorted(
        (
            entry["interrupter"],
            entry["interrupted"],
            float(entry["at"]),
            float(entry["overlap_sec"]),
        )
        for entry in data
    )


def test_overlap_matches_legacy_for_dense_turns() -> None:
    segments = _dense_turn_segments(speakers=3, turns_per_speaker=10)
    expected = _legacy_overlap(deepcopy(segments))
    result = compute_overlap_and_interruptions(deepcopy(segments))

    assert math.isclose(result["overlap_total_sec"], expected["overlap_total_sec"], rel_tol=1e-9)
    assert math.isclose(result["overlap_ratio"], expected["overlap_ratio"], rel_tol=1e-9)

    for speaker in set(result["by_speaker"]) | set(expected["by_speaker"]):
        res_slot = result["by_speaker"].get(speaker, {"overlap_sec": 0.0, "interruptions": 0})
        exp_slot = expected["by_speaker"].get(speaker, {"overlap_sec": 0.0, "interruptions": 0})
        assert math.isclose(float(res_slot["overlap_sec"]), float(exp_slot["overlap_sec"]), rel_tol=1e-9)
        assert int(res_slot["interruptions"]) == int(exp_slot["interruptions"])

    assert _sort_interruptions(result["interruptions"]) == _sort_interruptions(expected["interruptions"])


def test_sweep_line_outperforms_legacy_on_dense_turns() -> None:
    segments = _dense_turn_segments(
        speakers=5,
        turns_per_speaker=45,
        base_gap=0.02,
        duration=1.1,
    )

    expected = _legacy_overlap(deepcopy(segments))
    result = compute_overlap_and_interruptions(deepcopy(segments))

    assert math.isclose(
        result["overlap_total_sec"], expected["overlap_total_sec"], rel_tol=1e-9
    )

    legacy_times: list[float] = []
    optimized_times: list[float] = []

    for _ in range(5):
        start = time.perf_counter()
        _legacy_overlap(deepcopy(segments))
        legacy_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        compute_overlap_and_interruptions(deepcopy(segments))
        optimized_times.append(time.perf_counter() - start)

    assert median(optimized_times) <= median(legacy_times) * 0.75

