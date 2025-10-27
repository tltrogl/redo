#!/usr/bin/env python
"""Generate a polished HTML executive summary from DiaRemot outputs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

VQ_THRESHOLDS = {
    "vq_jitter_pct": 1.0,
    "vq_shimmer_db": 6.0,
    "vq_hnr_db": 15.0,
    "vq_cpps_db": 10.0,
    "vq_voiced_ratio": 0.5,
}
LOWER_IS_RISK = {"vq_hnr_db", "vq_cpps_db", "vq_voiced_ratio"}
VQ_METRICS = [
    "vq_jitter_pct",
    "vq_shimmer_db",
    "vq_hnr_db",
    "vq_cpps_db",
    "vq_voiced_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Produce enhanced HTML summary")
    parser.add_argument("segments_csv", type=Path, help="Path to diarized_transcript_with_emotion.csv")
    parser.add_argument("speakers_csv", type=Path, help="Path to speakers_summary.csv")
    parser.add_argument("--qc", type=Path, help="Optional qc_report.json for metadata")
    parser.add_argument("--output", type=Path, default=Path("enhanced_summary.html"))
    return parser.parse_args()


def _fnum(val: float | None, digits: int = 2) -> str:
    if val is None:
        return "—"
    return f"{val:.{digits}f}"


def load_speakers(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_segments(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Segments CSV not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def summarize_voice_quality(segments: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stats: dict[str, dict[str, float]] = {}
    alerts: list[dict[str, Any]] = []
    for metric in VQ_METRICS:
        values = [float(seg[metric]) for seg in segments if seg.get(metric)]
        if values:
            stats[metric] = {
                "avg": mean(values),
                "min": min(values),
                "max": max(values),
            }
    for seg in segments:
        speaker = seg.get("speaker_name") or seg.get("speaker_id") or "Unknown"
        start = float(seg.get("start") or 0.0)
        text = (seg.get("text") or "").strip()
        for metric, threshold in VQ_THRESHOLDS.items():
            val_raw = seg.get(metric)
            if not val_raw:
                continue
            value = float(val_raw)
            risky = value < threshold if metric in LOWER_IS_RISK else value > threshold
            if risky:
                alerts.append(
                    {
                        "speaker": speaker,
                        "time": start,
                        "metric": metric,
                        "value": value,
                        "threshold": threshold,
                        "direction": "<" if metric in LOWER_IS_RISK else ">",
                        "text": text[:160] + ("…" if len(text) > 160 else ""),
                    }
                )
    return stats, alerts[:20]


def key_moments(segments: list[dict[str, Any]], top_n: int = 5) -> list[dict[str, Any]]:
    scored = sorted(segments, key=lambda s: float(s.get("arousal") or 0.0), reverse=True)
    moments = []
    for seg in scored[:top_n * 3]:
        arousal = float(seg.get("arousal") or 0.0)
        if arousal < 0.35:
            continue
        moments.append(
            {
                "time": float(seg.get("start") or 0.0),
                "speaker": seg.get("speaker_name") or seg.get("speaker_id") or "Speaker",
                "arousal": arousal,
                "text": (seg.get("text") or "").strip(),
            }
        )
        if len(moments) >= top_n:
            break
    return moments


def speaker_headlines(speakers: list[dict[str, Any]]) -> list[str]:
    if not speakers:
        return []
    sorted_spk = sorted(speakers, key=lambda s: float(s.get("total_duration") or 0.0), reverse=True)
    headlines = []
    total_time = sum(float(s.get("total_duration") or 0.0) for s in speakers)
    for spk in sorted_spk[:3]:
        pct = 100 * float(spk.get("total_duration") or 0.0) / max(total_time, 1)
        name = spk.get("speaker_name") or f"Speaker {spk.get('speaker_id') or '?'}"
        wpm = spk.get("wpm_avg") or spk.get("wpm")
        headlines.append(f"{name} spoke {pct:.0f}% of the time (avg {float(wpm or 0):.0f} WPM).")
    return headlines


def render_html(*, segments: list[dict[str, Any]], speakers: list[dict[str, Any]], voice_stats: dict[str, Any], voice_alerts: list[dict[str, Any]], moments: list[dict[str, Any]], headlines: list[str], qc: dict[str, Any], out_path: Path) -> None:
    file_id = qc.get("file_id") or out_path.stem
    run_id = qc.get("run_id") or "N/A"
    duration = max((float(seg.get("end") or 0.0) for seg in segments), default=0.0)
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    def fmt_time(seconds: float) -> str:
        s = int(seconds)
        h, m = divmod(s, 3600)
        m, sec = divmod(m, 60)
        if h:
            return f"{h}h {m:02d}m {sec:02d}s"
        return f"{m}m {sec:02d}s"

    voice_rows = ""
    for metric in VQ_METRICS:
        stats = voice_stats.get(metric)
        if not stats:
            continue
        direction = "<" if metric in LOWER_IS_RISK else ">"
        threshold = VQ_THRESHOLDS.get(metric)
        voice_rows += (
            f"<tr><td>{metric}</td><td>{stats['avg']:.2f}</td><td>{stats['min']:.2f}</td><td>{stats['max']:.2f}</td><td>{direction} {threshold}</td></tr>"
        )

    alert_cards = "".join(
        f"""
        <div class=\"alert-card\">
          <strong>{a['speaker']} ({fmt_time(a['time'])})</strong><br/>
          {a['metric']} {a['direction']} {a['threshold']} (value={a['value']:.2f})<br/>
          <em>{a['text']}</em>
        </div>
        """
        for a in voice_alerts
    )

    moment_cards = "".join(
        f"""
        <div class=\"moment\">
          <div class=\"moment-time\">{fmt_time(m['time'])}</div>
          <div>
            <strong>{m['speaker']}</strong> — arousal {m['arousal']:.2f}<br/>
            <em>{m['text']}</em>
          </div>
        </div>
        """
        for m in moments
    ) or "<p>No high-arousal moments detected.</p>"

    speaker_cards = "".join(
        f"""
        <div class=\"speaker-card\">
          <h3>{spk.get('speaker_name') or f"Speaker {spk.get('speaker_id')}"}</h3>
          <p>Time: {fmt_time(float(spk.get('total_duration') or 0.0))}</p>
          <p>Avg WPM: {_fnum(float(spk.get('wpm_avg') or spk.get('wpm') or 0.0), 0)}</p>
          <p>Dominant emotion: {spk.get('dominant_text_emotion') or 'neutral'}</p>
        </div>
        """
        for spk in speakers[:4]
    ) or "<p>No speaker summary available.</p>"

    exec_items = "".join(f"<li>{line}</li>" for line in headlines) or "<li>No speaker highlights available.</li>"

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset='utf-8'/>
<title>Enhanced Summary - {file_id}</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; background:#f5f7fb; margin:0; color:#1f2a44; }}
.container {{ max-width: 960px; margin: auto; padding: 32px; }}
.section {{ background:#fff; border-radius:12px; padding:20px; margin-bottom:20px; box-shadow:0 4px 16px rgba(15,23,42,0.08); }}
.section h2 {{ margin-top:0; color:#1f2a44; }}
.alert-card {{ border-left:4px solid #e74c3c; padding:12px; margin-bottom:12px; background:#fff5f5; }}
.highlight-list {{ margin:0; padding-left:1.2rem; }}
.moment {{ border-left:4px solid #5b7cfa; padding-left:12px; margin-bottom:12px; }}
.moment-time {{ font-weight:bold; color:#5b7cfa; }}
.speaker-grid {{ display:flex; gap:16px; flex-wrap:wrap; }}
.speaker-card {{ flex:1; min-width:200px; background:#eef2ff; border-radius:10px; padding:16px; }}
table {{ width:100%; border-collapse:collapse; }}
th, td {{ padding:8px; border-bottom:1px solid #e0e6ff; text-align:left; }}
th {{ background:#eef2ff; }}
footer {{ text-align:center; font-size:0.85rem; color:#6c7a99; margin-top:32px; }}
</style>
</head>
<body>
<div class='container'>
  <div class='section'>
    <h1>Enhanced Conversation Summary</h1>
    <p><strong>File:</strong> {file_id} &nbsp; <strong>Run ID:</strong> {run_id} &nbsp; <strong>Duration:</strong> {fmt_time(duration)} &nbsp; <strong>Generated:</strong> {gen_time}</p>
  </div>
  <div class='section'>
    <h2>Executive Highlights</h2>
    <ul class='highlight-list'>
      {exec_items}
    </ul>
  </div>
  <div class='section'>
    <h2>Key Moments</h2>
    {moment_cards}
  </div>
  <div class='section'>
    <h2>Voice Quality Insights</h2>
    <table>
      <tr><th>Metric</th><th>Avg</th><th>Min</th><th>Max</th><th>Threshold</th></tr>
      {voice_rows or '<tr><td colspan="5">No voice metrics available</td></tr>'}
    </table>
    {alert_cards or '<p>No voice-quality alerts detected.</p>'}
  </div>
  <div class='section'>
    <h2>Speaker Snapshots</h2>
    <div class='speaker-grid'>
      {speaker_cards}
    </div>
  </div>
</div>
<footer>Generated by DiaRemot enhanced summary script.</footer>
</body></html>
"""

    out_path.write_text(html, encoding="utf-8")
    print(f"Enhanced summary written to {out_path}")


def main() -> None:
    args = parse_args()
    segments = load_segments(args.segments_csv)
    speakers = load_speakers(args.speakers_csv)
    qc = json.loads(args.qc.read_text()) if args.qc and args.qc.exists() else {}
    voice_stats, voice_alerts = summarize_voice_quality(segments)
    moments = key_moments(segments)
    headlines = speaker_headlines(speakers)
    render_html(
        segments=segments,
        speakers=speakers,
        voice_stats=voice_stats,
        voice_alerts=voice_alerts,
        moments=moments,
        headlines=headlines,
        qc=qc,
        out_path=args.output,
    )


if __name__ == "__main__":
    main()
