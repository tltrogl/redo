# Sample1 Pipeline Run — 2025-10-29

This record captures the end-to-end DiaRemot pipeline execution against `data/sample1.mp3`.

## Preparation

- Ensured the venv was active and exported the canonical cache/model environment variables outlined in `AGENTS.md`.
- Verified the downloaded model bundle checksum (`33d0a9194de3cbd667fd329ab7c6ce832f4e1373ba0b1844ce0040a191abc483`) before extracting it into `./models/`.
- Normalised the Silero VAD directory name to `Diarization/silero_vad/` to match the expected loader alias.

> **Note:** The repository does not ship `data/sample1.mp3`; for this run the file was provisioned by copying the bundled `data/sample2.mp3` into `data/sample1.mp3` so the CLI could operate on the expected sample name.

## Command

```
PYTHONPATH=src python -m diaremot.cli run \
  --input data/sample1.mp3 \
  --outdir outputs/sample1_run \
  --model-root "$DIAREMOT_MODEL_DIR" \
  --affect-backend onnx \
  --sed-mode auto
```

## Stage Results

| Stage | Status | Wall Time |
|-------|--------|-----------|
| dependency_check | PASS | 00:00.000 |
| preprocess | PASS | 00:31.298 |
| background_sed | PASS | 00:02.072 |
| diarize | PASS | 00:05.456 |
| transcribe | PASS | 00:58.598 |
| paralinguistics | PASS | 00:09.866 |
| affect_and_assemble | PASS | 00:49.093 |
| overlap_interruptions | PASS | 00:00.000 |
| conversation_analysis | PASS | 00:00.001 |
| speaker_rollups | PASS | 00:00.001 |
| outputs | PASS | 00:00.014 |

The Faster-Whisper tiny.en export from the bundled `models/` directory powered transcription with `compute_type=int8`. SED, affect, intent, text-emotion, and speaker analytics all ran in ONNX mode without triggering fallbacks.

## Key Outputs

All primary artefacts were generated under `outputs/sample1_run/`:

- `diarized_transcript_with_emotion.csv`
- `diarized_transcript_readable.txt`
- `segments.jsonl`
- `speakers_summary.csv`
- `timeline.csv`
- `summary.html`
- `summary.pdf`
- `qc_report.json`

The QC report produced no warnings and recorded the full stage timing breakdown and dependency health metadata.
