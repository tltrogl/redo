#!/usr/bin/env python
"""Summarize DiaRemot voice-quality metrics and surface risk flags."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

THRESHOLDS = {
    "vq_jitter_pct": 1.0,          # >1% jitter often indicates instability
    "vq_shimmer_db": 6.0,          # >6 dB shimmer considered elevated
    "vq_hnr_db": 15.0,             # <15 dB can indicate roughness
    "vq_cpps_db": 10.0,            # <10 dB CPPS associated with breathiness
    "vq_voiced_ratio": 0.5,        # <0.5 voiced coverage may indicate weak signal
}

LOWER_IS_RISK = {"vq_hnr_db", "vq_cpps_db", "vq_voiced_ratio"}

METRICS = [
    "vq_jitter_pct",
    "vq_shimmer_db",
    "vq_hnr_db",
    "vq_cpps_db",
    "vq_voiced_ratio",
    "vq_spectral_slope_db",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze DiaRemot diarized_transcript_with_emotion.csv for voice-quality risks."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to diarized_transcript_with_emotion.csv",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="How many flagged segments per speaker to show (default: 5)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional path to write JSON summary",
    )
    return parser.parse_args()


def _to_float(value: Any) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if val != val:  # NaN
        return None
    return val


def main() -> None:
    args = parse_args()
    if not args.csv_path.exists():
        raise SystemExit(f"CSV file not found: {args.csv_path}")

    per_speaker: dict[str, dict[str, list[tuple[float, dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    aggregations: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    flagged_segments: list[dict[str, Any]] = []

    with args.csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [m for m in METRICS if m not in reader.fieldnames]
        if missing:
            raise SystemExit(f"CSV missing required columns: {missing}")
        for row in reader:
            speaker = row.get("speaker_name") or row.get("speaker_id") or "unknown"
            for metric in METRICS:
                val = _to_float(row.get(metric))
                if val is None:
                    continue
                aggregations[speaker][metric].append(val)
            # risk detection
            segment_flagged = False
            reasons: list[str] = []
            for metric, threshold in THRESHOLDS.items():
                val = _to_float(row.get(metric))
                if val is None:
                    continue
                risk = val < threshold if metric in LOWER_IS_RISK else val > threshold
                if risk:
                    segment_flagged = True
                    comparator = "<" if metric in LOWER_IS_RISK else ">"
                    reasons.append(f"{metric} {comparator} {threshold} (value={val:.2f})")
                    per_speaker[speaker][metric].append((val, row))
            if segment_flagged:
                flagged_segments.append(
                    {
                        "speaker": speaker,
                        "start": _to_float(row.get("start")),
                        "end": _to_float(row.get("end")),
                        "text": row.get("text", ""),
                        "reasons": reasons,
                    }
                )

    summary: dict[str, Any] = {"speakers": {}, "flagged_segments": flagged_segments}

    for speaker, metrics in aggregations.items():
        summary["speakers"][speaker] = {}
        for metric, values in metrics.items():
            summary["speakers"][speaker][metric] = {
                "count": len(values),
                "avg": mean(values),
                "min": min(values),
                "max": max(values),
            }

    # Console report
    print("Voice Quality Summary\n=====================")
    if not summary["speakers"]:
        print("No voice-quality metrics found (did paralinguistics run?)")
    for speaker, metrics in summary["speakers"].items():
        print(f"\nSpeaker: {speaker}")
        for metric in METRICS:
            stats = metrics.get(metric)
            if not stats:
                continue
            print(
                f"  {metric}: avg={stats['avg']:.2f} min={stats['min']:.2f} max={stats['max']:.2f}"
            )
        if speaker in per_speaker:
            print("  Flagged segments:")
            for metric, entries in per_speaker[speaker].items():
                sorted_entries = sorted(
                    entries,
                    key=lambda item: item[0],
                    reverse=(metric not in LOWER_IS_RISK),
                )[: args.top]
                for val, row in sorted_entries:
                    start = row.get("start")
                    end = row.get("end")
                    text = (row.get("text") or "").strip()
                    short_text = text[:80] + ("…" if len(text) > 80 else "")
                    comparator = ">" if metric not in LOWER_IS_RISK else "<"
                    print(
                        f"    {metric} {comparator} {THRESHOLDS[metric]} (value={val:.2f}) at {start}-{end}: {short_text}"
                    )

    if args.json:
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nJSON summary written to {args.json}")


if __name__ == "__main__":
    main()
