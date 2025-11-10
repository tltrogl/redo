from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from diaremot.summaries.speakers_summary_builder import build_speakers_summary


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "outputs_smoke"


def _load_segments(path: Path) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            segments.append(json.loads(line))
    return segments


def _render_csv(rows: list[dict[str, object]], header: list[str]) -> list[str]:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=header)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in header})
    return buffer.getvalue().strip().splitlines()


def test_speakers_summary_matches_snapshot() -> None:
    segments = _load_segments(FIXTURE_ROOT / "segments.jsonl")
    summary = build_speakers_summary(segments, {})

    csv_path = FIXTURE_ROOT / "speakers_summary.csv"
    expected_lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    reader = csv.DictReader(io.StringIO("\n".join(expected_lines)))
    header = reader.fieldnames or []
    assert header, "snapshot CSV missing header"

    actual_lines = _render_csv(summary, header)
    assert actual_lines == expected_lines


def test_interruptions_payload_overrides_counts() -> None:
    segments = [
        {
            "speaker_id": "S1",
            "speaker_name": "Speaker 1",
            "start": 0.0,
            "end": 10.0,
            "text": "hello world",
            "words": 20,
        }
    ]
    per_speaker_interrupts = {
        "S1": {
            "made": 3,
            "received": 2,
            "overlap_sec": 1.5,
        }
    }

    summary = build_speakers_summary(segments, per_speaker_interrupts)
    assert len(summary) == 1
    speaker = summary[0]
    assert speaker["interruptions_made"] == 3
    assert speaker["interruptions_received"] == 2
    # Overlap seconds / total duration -> ratio
    assert speaker["overlap_ratio"] == 0.15
