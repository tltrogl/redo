import csv
from pathlib import Path
from statistics import mean
from collections import Counter, defaultdict

from diaremot.summaries.conversation_analysis import analyze_conversation_flow
from diaremot.summaries.narrative_builder import build_narrative
from diaremot.pipeline.outputs import write_narrative_report
from diaremot.summaries.html_summary_generator import HTMLSummaryGenerator
from diaremot.summaries.pdf_summary_generator import PDFSummaryGenerator

base = Path(r'D:\Audio\outs\troglin')
transcript_path = base / 'diarized_transcript_with_emotion.csv'
speakers_summary_path = base / 'speakers_summary.csv'
if not transcript_path.exists():
    raise SystemExit(f'Missing transcript CSV: {transcript_path}')
if not speakers_summary_path.exists():
    raise SystemExit(f'Missing speakers summary CSV: {speakers_summary_path}')

segments = []
with transcript_path.open(newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            start = float(row.get('start') or 0.0)
            end = float(row.get('end') or 0.0)
        except ValueError:
            continue
        duration = float(row.get('duration_s') or (end - start) or 0.0)
        segments.append(
            {
                'file_id': row.get('file_id') or 'audio',
                'start': start,
                'end': end,
                'duration_s': duration,
                'speaker_id': (row.get('speaker_id') or 'Unknown').strip(),
                'speaker_name': (row.get('speaker_name') or row.get('speaker_id') or 'Unknown').strip(),
                'text': (row.get('text') or '').strip(),
                'valence': float(row.get('valence') or 0.0),
                'arousal': float(row.get('arousal') or 0.0),
                'dominance': float(row.get('dominance') or 0.0),
                'emotion_top': (row.get('emotion_top') or '').strip(),
                'emotion_scores_json': row.get('emotion_scores_json') or '',
                'text_emotions_top5_json': row.get('text_emotions_top5_json') or '',
                'text_emotions_full_json': row.get('text_emotions_full_json') or '',
                'intent_top': (row.get('intent_top') or '').strip(),
                'intent_top3_json': row.get('intent_top3_json') or '',
                'events_top3_json': row.get('events_top3_json') or '',
                'noise_tag': row.get('noise_tag') or '',
                'asr_logprob_avg': row.get('asr_logprob_avg'),
                'snr_db': row.get('snr_db'),
                'snr_db_sed': row.get('snr_db_sed'),
                'wpm': float(row.get('wpm') or 0.0),
                'duration': duration,
            }
        )

if not segments:
    raise SystemExit('No segments parsed from transcript CSV')

total_duration = max(seg['end'] for seg in segments)
num_segments = len(segments)

speaker_durations = defaultdict(float)
for seg in segments:
    key = (seg['speaker_id'], seg['speaker_name'])
    speaker_durations[key] += seg['duration']

segments_for_metrics = [
    {
        'speaker_id': seg['speaker_id'],
        'speaker': seg['speaker_id'],
        'start': seg['start'],
        'end': seg['end'],
        'text': seg['text'],
    }
    for seg in segments
]

conv_metrics = analyze_conversation_flow(segments_for_metrics, total_duration)

with speakers_summary_path.open(newline='', encoding='utf-8') as f:
    speaker_reader = csv.DictReader(f)
    speakers_summary = [dict(row) for row in speaker_reader]

narrative = build_narrative(segments, speakers_summary, conv_metrics, total_duration=total_duration)

write_narrative_report(base / 'conversation_report.md', narrative)

html_gen = HTMLSummaryGenerator()
html_gen.render_to_html(
    out_dir=str(base),
    file_id='troglin',
    segments=segments,
    speakers_summary=speakers_summary,
    overlap_stats={},
    narrative=narrative,
)

pdf_gen = PDFSummaryGenerator()
pdf_gen.render_to_pdf(
    out_dir=str(base),
    file_id='troglin',
    segments=segments,
    speakers_summary=speakers_summary,
    overlap_stats={},
    narrative=narrative,
)

print('Narrative outputs regenerated in', base)
