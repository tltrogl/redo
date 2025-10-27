This file is for the agent. Execute deterministic instructions exactly as written. If a prerequisite is missing or a contract below would be violated, stop and return a failure report with the minimal relevant logs.
Agents read AGENTS.md to learn build/test/style rules and context. Think of it as a README for agents. 

Your core function is efficient and safe assistance. Balance extreme conciseness with the crucial need for clarity, especially regarding safety and potential system modifications. Always prioritize user control and project conventions. Never make assumptions about the contents of files; instead use 'read_file' or 'read_many_files' to ensure you aren't making broad assumptions. Finally, you are an agent - please keep going until the user's query is completely resolved.


0) Operating Mode


Network: Internet ON. 

Execution: CPU-only (no CUDA paths).

Inference policy: ONNX-first

1) Project Maps (fast mental model)
1.1 Pipeline Stage Map (fixed 11 stages)
1 dependency_check
2 preprocess
3 background_sed
4 diarize
5 transcribe
6 paralinguistics
7 affect_and_assemble
8 overlap_interruptions
9 conversation_analysis
10 speaker_rollups
11 outputs

auto_tune.py is internal to diarization and not an extra stage.

1.2 Dataflow Map (11-stage pipeline → outputs)
See DATAFLOW.md for detailed stage-by-stage documentation.

Audio (1–3h)
    ↓
[1] dependency_check → Validate environment
    ↓
[2] preprocess → 16kHz mono, -20 LUFS, denoise, auto-chunk
    ↓ {y, sr, duration_s, health, audio_sha16}
    ↓
[3] background_sed → PANNs CNN14 (global + timeline if noisy)
    ↓ {sed_info: top labels, dominant_label, noise_score}
    ↓
[4] diarize → Silero VAD + ECAPA embeddings + AHC clustering
    ↓ {turns: [{start, end, speaker, speaker_name}]}
    ↓
[5] transcribe → Faster-Whisper with intelligent batching
    ↓ {norm_tx: [{start, end, speaker, text, asr_logprob_avg}]}
    ↓
[6] paralinguistics → Praat (jitter/shimmer/HNR/CPPS) + prosody
    ↓ {para_map: {seg_idx: {wpm, vq_*, f0_*, ...}}}
    ↓
[7] affect_and_assemble → Audio (VAD+SER8) + Text (GoEmotions+BART-MNLI)
    ↓ {segments_final: 39 columns per segment}
    ↓
[8] overlap_interruptions → Overlaps + interruption classification
    ↓ {overlap_stats, per_speaker_interrupts}
    ↓
[9] conversation_analysis → Turn-taking + dominance + flow
    ↓ {conv_metrics}
    ↓
[10] speaker_rollups → Per-speaker aggregated stats
    ↓ {speakers_summary}
    ↓
[11] outputs → CSV/JSONL/HTML/PDF/QC reports


Artifacts:

Primary CSV: diarized_transcript_with_emotion.csv (39 columns, fixed order per SEGMENT_COLUMNS).

Other defaults: segments.jsonl, speakers_summary.csv, events_timeline.csv, summary.html, summary.pdf, qc_report.json, timeline.csv, diarized_transcript_readable.txt, speaker_registry.json.

Cache files: .cache/{audio_sha16}/preprocessed.npz, diar.json, tx.json (enables resume).

1.3 Model Map (preferred ONNX; CPU-friendly)
VAD	Silero VAD (ONNX)	Well-known CPU VAD; ONNX variants exist. 

Speaker embeds	ECAPA-TDNN (ONNX if present)	Standard embeddings for AHC clustering.

SED	PANNs CNN14 (ONNX) → AudioSet→~20 groups	CNN14 summary + paper. 

ASR	Faster-Whisper on CTranslate2	Fast CPU inference; int8 default; float32 remains optional here. 

Tone (V/A/D)	wav2vec2-based (per brief)	—
SER (8-class)	wav2vec2 emotion model	—
Text emotions	RoBERTa GoEmotions (28)	
Intent	BART-large-MNLI (zero-shot)	

Runtime	ONNX Runtime CPU EP	CPU EP is default provider. 
ONNX Runtime

1.4 Directory Map (logical anchors)

Stages registry: src/diaremot/pipeline/stages/__init__.py (defines PIPELINE_STAGES)

Outputs schema: src/diaremot/pipeline/outputs.py (defines SEGMENT_COLUMNS)

Orchestrator/CLI: diaremot/cli.py (python -m diaremot.cli run), diaremot/pipeline/run_pipeline.py, legacy diaremot/pipeline/cli_entry.py

Speaker registry persistence: e.g., speaker_registry.json (centroids, names)


3) Deterministic Task Protocol (what you return every time)

A) Plan
Name exact files/symbols and the one-clause reason for each edit.

B) Code Changes
Apply surgical and thorough unified diffs limited to scope; preserve style/imports/APIs. Emit per-file diffs.

C) Verification Gates (run all, capture exit codes + trimmed logs)

# Lint
ruff check src/ tests/

# Unit tests
pytest -q

# Smoke (offline-safe after models are installed)
python -m diaremot.cli run -i data\sample.wav -o outputs\_smoke --disable-affect --disable-sed

# Contracts (hard fail if violated)
python - << 'PY'
from diaremot.pipeline.outputs import SEGMENT_COLUMNS
assert len(SEGMENT_COLUMNS)==40, f"CSV columns != 39 ({len(SEGMENT_COLUMNS)})"
print("CSV schema OK")
PY

python - << 'PY'
from diaremot.pipeline.stages import PIPELINE_STAGES
assert len(PIPELINE_STAGES)==11, f"Stage count != 11 ({len(PIPELINE_STAGES)})"
print("Stage count OK")
PY

python - << 'PY'
import os
v=os.environ.get("CUDA_VISIBLE_DEVICES","")
assert v in ("","none","None"), f"Unexpected CUDA_VISIBLE_DEVICES={v!r}"
print("CPU-only OK")
PY


Notes: ONNX Runtime’s CPU EP is the baseline; we intentionally avoid GPU providers. 
ONNX Runtime

D) Report (structured)

Summary (1–3 sentences)

Files Modified (path + +/- counts or tiny diff excerpts)

Commands Executed (verbatim + exit codes)

Key Logs (failing tails or final summaries only)

Risks / Assumptions (bullets)

Follow-Up (optional)

4) DiaRemot Contracts (enforce)

CSV schema: diarized_transcript_with_emotion.csv has exactly 39 columns, fixed order. Never remove/reorder; append only with migration + tests + docs.

Stage count: Exactly 11 (map above), order fixed.

CPU-only: no CUDA/GPU paths.

ONNX-first: Prefer ONNXRuntime; log explicit fallbacks. 
ONNX Runtime

Defaults to preserve:

ASR: Faster-Whisper tiny.en via CTranslate2; int8 default in the main pipeline (float32 only when explicitly requested or ASR-only subcommand). 
GitHub

SED: PANNs CNN14 ONNX; 1.0 s frames / 0.5 s hop; median 3–5; hysteresis enter ≥0.50 / exit ≤0.35; min_dur=0.30 s; merge_gap≤0.20 s. 

Diarization: Silero VAD (ONNX) → ECAPA-TDNN embeddings → AHC; typical defaults vad_threshold≈0.35, vad_min_speech_sec=0.80, vad_min_silence_sec=0.80, vad_speech_pad_sec=0.10, ahc_distance_threshold≈0.15; post rules: collar≈0.25 s, min_turn_sec=1.50, max_gap_to_merge_sec=1.00. 
Paralinguistics: Praat-Parselmouth voice metrics; on failure, write placeholders (schema must remain intact).

NLP: GoEmotions (28) and BART-MNLI intent; keep JSON distributions. 

5) CLI Surfaces (for smoke / automation)

Primary: python -m diaremot.cli run (Typer). Defaults: all stages on; ASR int8.

Direct: python -m diaremot.pipeline.run_pipeline (explicit config/env).

Legacy: python -m diaremot.pipeline.cli_entry (ASR default may differ).

PowerShell quick smoke:

python -m diaremot.cli run -i data\sample.wav -o outputs\_smoke --disable-affect --disable-sed

6) CSV Schema (39 columns — names & categories)

Temporal: file_id, start, end, duration_s
Speaker: speaker_id, speaker_name
Content: text, words, language
ASR Confidence: asr_logprob_avg, low_confidence_ser
Audio Emotion / Tone: valence, arousal, dominance, emotion_top, emotion_scores_json
Text Emotions: text_emotions_top5_json, text_emotions_full_json
Intent: intent_top, intent_top3_json
SED: events_top3_json, noise_tag
Voice Quality (Praat): vq_jitter_pct, vq_shimmer_db, vq_hnr_db, vq_cpps_db, voice_quality_hint
Prosody: wpm, pause_count, pause_time_s, pause_ratio, f0_mean_hz, f0_std_hz, loudness_rms, disfluency_count
Signal Quality: snr_db, snr_db_sed, vad_unstable
Hints/Flags: affect_hint, error_flags

7) Style & Scope

Surgical but thorough diffs only.

Modify files inside the repo unless directed otherwise.

Maintain structured logging; surface failures (do not drop required columns).

8) Failure Policy

If any gate fails or a prerequisite is missing, stop and return:

Failing command + exit code

Last ~20 log lines (load-bearing tail)

One-paragraph diagnosis

References (for maintainers; safe to keep at bottom)

AGENTS.md concept/spec (repo + site). Agents read this file to learn environment/setup/tests. 


Release (models.zip) — tltrogl/diaremot2-ai v2.AI with asset for Codex setup.
Codex Cloud provisioning always fetches `models.zip` from this release; keep it
current and verify the checksum before promoting new assets.

### Codex Cloud bootstrap checklist (must pass before promotion)

1. **Fetch models bundle**
   ```bash
   curl -L https://github.com/tltrogl/diaremot2-ai/releases/download/v2.AI/models.zip -o models.zip
   sha256sum --check models.zip.sha256  # expects 3cc2115f4ef7cd4f9e43cfcec376bf56ea2a8213cb760ab17b27edbc2cac206c
   unzip -q models.zip -d ./models
   ```
   The archive expands into the alias-aware layout (`goemotions-onnx/`, `ser8-onnx/`,
   `panns/`, `bart/`, `ecapa_onnx/`, root `silero_vad.onnx`). The bundle currently
   omits the dimensional VAD model; the pipeline will emit neutral placeholders unless
   an `affect/vad_dim/` directory is provided manually from an internal export.

2. **Install the pinned runtime**
   ```bash
   pip install -r requirements.txt
   ```
   This installs the ONNX-first stack plus `transformers==4.44.2`, `tokenizers==0.19.1`,
   and all paralinguistics/audio dependencies expected by the smoke test.

3. **Run the CPU smoke test with affect enabled**
   ```bash
   PYTHONPATH=src \
     HF_HOME=./.cache \
     python -m diaremot.cli smoke \
       --outdir /tmp/smoke_test \
       --model-root ./models \
       --enable-affect
   ```
   First run downloads Faster-Whisper tiny.en (CTranslate2) and may contact
   Hugging Face to hydrate tokenizer metadata for BART if the local JSON bundle
   fails validation. Subsequent runs stay offline. Treat missing optional VAD as
   a warning, not a failure, until the upstream release ships the asset.

4. **Verify outputs**
   Ensure `/tmp/smoke_test/` contains the standard CSV/JSON/HTML/PDF artefacts and
   that the stage summary reports `PASS` for all 11 stages. Investigate any
   `issues` recorded by `affect` before shipping.

ONNX Runtime EPs (CPU) — CPU EP is the default; we stay CPU-only. 
ONNX Runtime

Faster-Whisper / CTranslate2 — CPU-optimized Whisper inference. 

Silero VAD (ONNX variants exist). 

ECAPA-TDNN embeddings (speaker verification). 
Hugging Face

PANNs CNN14 (AudioSet-trained SED). 

GoEmotions (RoBERTa base). 

FacebookAI/roberta-large-mnli 

read readme.md
read gemini.md