# AGENTS.md — Local IDE Agent Playbook (Technical)

This file defines how the interactive IDE agent operates on this Windows workstation for DiaRemot. It is specific to the local, human‑in‑the‑loop IDE flow and must not be confused with the non‑interactive cloud agent guidance in `agentscloud.md`.

---

## 1. Scope & Role
- Operate in `D:\dia\diaremot2-on` with the existing venv and models.
- Explain actions before executing, then perform the smallest most thorough change that achieves the goal.
- Prefer reading artefacts (logs, caches, outputs) to re‑running heavy stages; only re‑run when inputs/config changed or artefacts are missing/invalid. Never assume always verify.
- Keep output concise; include exact commands, exit codes, and paths for anything you run or change.

---

## 2. Guardrails
- Plan First: briefly outline intent, edits/tests, and expected outcome before touching files. Clearly present plans for approval before implementing.
- Stay Surgical: modify only files relevant to the plan; avoid drive‑by refactors.
- Evidence: after each task, report commands, exit codes, and the specific artefacts you inspected.
- Reproducibility: do not clear caches or models without stating why and what will be lost; back up outputs first.
- Complete: see all tasks through to completion. If sidetracked or interrupted always return and complete the task.
- One Step: when giving multi step directions to user give one step, wait for completetion, then give next step.
- Separation: this file governs the IDE agent only. Cloud automation rules live in `agentscloud.md`.

---

## 3. Environment Facts (Local)
- Repository root: `D:\dia\diaremot2-on`
- Virtualenv: `.balls` (PowerShell activation: `.\\.balls\\Scripts\\Activate.ps1`)
- Models: primary at `D:\models`. If unavailable, use `./assets/models.zip` (must be the approved bundle) and extract to `./models`.
- Caches & outputs: Hugging Face/CT2 under `.cache\\`; checkpoints under `checkpoints\\`; recent runs under `outputs_*` or `audio\\outs\\<name>`.
- Sample audio: `data\\sample1.mp3`, `data\\sample2.mp3`.

---

## 4. Working Sequence (IDE)
1) Context Check
   - `git status -sb`
   - Inspect latest outputs and `logs\\run.jsonl`
   - Note any “fallback”, “warn”, or long‑running stages
2) Plan & Confirm
   - Write a short plan (goal → edits/tests → expected results)
3) Execute
   - Activate venv: `.\\.balls\\Scripts\\Activate.ps1`
   - Set env for caches/models:
     ```powershell
     $env:HF_HOME = (Resolve-Path '.\\.cache')
     $env:TRANSFORMERS_CACHE = $env:HF_HOME
     $env:DIAREMOT_MODEL_DIR = 'D:\\models'
     ```
   - Run only the necessary pipeline commands
4) Verify
   - Validate against artefacts listed in Section 6
5) Summarise
   - What changed; commands + exit codes; artefact highlights; next steps

---

## 5. Canonical Commands (Local)
- Activate venv
  ```powershell
  .\.balls\Scripts\Activate.ps1
  ```
- Full run (default outdir: <input parent>\outs\<input stem>)
  ```powershell
  # Python module entry
  python -m diaremot.cli run "audio\sample1.mp3"
  # Or console script entry
  diaremot run "audio\sample1.mp3"
  ```
- Resume using caches/checkpoints
  ```powershell
  # Python module entry
  python -m diaremot.cli resume --input "audio\sample1.mp3"
  # Or console script entry
  diaremot resume --input "audio\sample1.mp3"
  ```
- Smoke test (synthetic)
  ```powershell
  python -m diaremot.cli smoke --enable-affect --model-root $env:DIAREMOT_MODEL_DIR
  ```
- Lint & tests
  ```powershell
  ruff check src/ tests/
  pytest -q
  ```

Notes
- When `--outdir` is omitted, outputs default to `<input parent>\\outs\\<input stem>`.
- Use `--clear-cache` only when a fresh run is required; prefer clearing specific cached files.

---

## 6. What To Inspect (Before Re‑runs)
- `logs\\run.jsonl`: stage timings, speaker estimates, warnings/fallbacks
- `outputs/*/qc_report.json`: `dependency_ok`, `warnings`, and `config_snapshot`
- `diarized_transcript_with_emotion.csv`: must have the full 39‑column schema
- `speakers_summary.csv`: check estimated speaker count vs expectations
- `segments.jsonl`: ensure V/A/D, text emotions, and intent fields are populated
- `summary.html` / `summary.pdf`: confirm report generation (missing PDF often indicates wkhtmltopdf absent)

Document anomalies even if you don’t fix them immediately.

---

## 7. Cache & Output Handling
- Back up outputs before clearing:
  ```powershell
  Copy-Item audio\\outs\\sample1 audio\\outs\\sample1_cached -Recurse
  ```
- Targeted cache clears (prefer these over global wipes):
  ```powershell
  # Clear diarization/transcription cache entries only
  Get-ChildItem .cache -Recurse -Include diar.json,tx.json | Remove-Item -Force
  ```
- Global cache clear (avoid unless necessary):
  ```powershell
  Remove-Item -Recurse -Force .cache, checkpoints
  ```

---

## 8. Models Policy (IDE)
- Primary source: `D:\\models`.
- Fallback: `./assets/models.zip` (organisation‑approved bundle). Extract to `./models` and set `DIAREMOT_MODEL_DIR` accordingly.
- Do not download alternative model files without an explicit reason documented in your summary.

---

## 9. Documentation Cross‑Checks
- When behaviour seems off, cross‑validate `README.md`, `DATAFLOW.md`, and `MODEL_MAP.md` against current code and artefacts.
- If a doc is wrong or incomplete, note the discrepancy in your summary and suggest a precise fix (file + line reference).

---

## 10. Completion Checklist
- Provide in your final message:
  - Summary of changes and rationale
  - Commands and exit codes
  - Artefact paths inspected and key observations
  - Follow‑up recommendations (if any)
- Leave the workspace ready for the next task (no stray temp files).

---

This technical playbook keeps the IDE session predictable, evidences every action, and reduces unnecessary re‑runs while staying aligned with how the local pipeline actually behaves.

---

## 11. Operational TODOs (DiaRemot v2.2.0)

Use this checklist to drive the pipeline to a stable, fast state on both local and VM environments.

For detailed step-by-step instructions (commands, verification, rollback), see `docs/TODO.md`.

### Critical Issues (Fix First)
- Silero VAD pathing: ensure ONNX loads. Prefer env var over discovery.
  - `SILERO_VAD_TORCH=0`
  - `SILERO_VAD_ONNX_PATH=/home/<user>/dia/diaremot2-on/models/diarization/silaro_vad/silero_vad.onnx`
- Diarization over‑segmentation: tighten VAD + bound clustering.
  - `--vad-backend onnx --vad-threshold 0.22 --vad-min-speech-sec 0.25 --vad-min-silence-sec 0.25 --vad-speech-pad-sec 0.20`
  - `--speaker_limit 4` (start; adjust 3–6 as needed)
  - Keep `ahc_distance_threshold ≈ 0.12–0.16`, `min_turn_sec ≥ 1.2s`, `max_gap_to_merge_sec ≈ 1.0s`, post‑merge distance ≥ 0.30.
- Preprocess cache crash: `_load_diar_tx_caches` must not call `guard.progress` (out of scope). Use pipeline logger instead. File: `src/diaremot/pipeline/stages/preprocess.py`.
- Report correctness: count unique speakers by label; dominance from summed per‑speaker `total_duration`. Files:
  - `src/diaremot/summaries/narrative_builder.py`
  - `src/diaremot/summaries/html_summary_generator.py`
  - `src/diaremot/summaries/pdf_summary_generator.py`

### VM Performance Guardrails
- Prevent hidden BLAS thread storms; let the app use cores:
  - `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1`
- Faster ECAPA on CPU: `DIAREMOT_ECAPA_MAX_BATCH=1024` (tune to RAM).
- ASR threads: `--asr-cpu-threads 4..8` (auto‑pick from `nproc`).

### Two‑Phase Flow (Recommended)
- Phase A (Core): dependency_check → preprocess → diarize → transcribe → minimal outputs.
  - Command: `diaremot ... --no-enable-sed --disable-affect`
- Phase B (Enrichment): paralinguistics → affect → SED → reports (from Phase A artifacts).
  - Command: `diaremot ... --enable-sed` (omit `--disable-affect`)

### CLI Ergonomics
- Expose clustering controls in CLI (if missing on a machine):
  - `--speaker_limit`, `--min-speakers`, `--max-speakers`, `--clustering-backend {ahc,spectral}`.

### Standardize Preprocess Defaults
- Use consistent flags across runs to avoid cache churn (`pp_signature`).
- Defaults: noise reduction ON; loudness mode `asr`.

### Sanity Checks (Add to QC / Logs)
- Warn if `speakers_est` ≫ unique speakers (e.g., est>50 and unique<10).
- Validate `diarized_transcript_with_emotion.csv` schema is 39 cols.
- Fail‑soft with hint if Silero ONNX not found (print `SILERO_VAD_ONNX_PATH`).

### Runbook Snippets
- Targeted cache clear by audio SHA (keeps models):
  ```bash
  python - << 'PY'
  import hashlib, pathlib, shutil
  a=pathlib.Path('audio/yourfile.ext'); h=hashlib.blake2s(digest_size=16)
  with a.open('rb') as f:
      for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
  d=pathlib.Path('.cache')/h.hexdigest(); print('AUDIO_SHA16=',h.hexdigest())
  if d.exists(): shutil.rmtree(d, ignore_errors=True)
  PY
  ```
- “VM perf” core run (reuse preprocess/ASR; redo diar only):
  ```bash
  export HF_HOME=.cache TRANSFORMERS_CACHE=.cache
  export DIAREMOT_MODEL_DIR=/home/<user>/dia/diaremot2-on/models
  export SILERO_VAD_TORCH=0
  export SILERO_VAD_ONNX_PATH=/home/<user>/dia/diaremot2-on/models/diarization/silaro_vad/silero_vad.onnx
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
  find .cache -maxdepth 3 -type f -name diar.json -delete
  PYTHONPATH=src python -m diaremot.pipeline.cli_entry \
    --input audio/yourfile.ext --outdir audio/outs/core \
    --vad-backend onnx --vad-threshold 0.22 --vad-min-speech-sec 0.25 \
    --vad-min-silence-sec 0.25 --vad-speech-pad-sec 0.20 --speaker_limit 4 \
    --no-enable-sed --disable-affect
  ```

### Acceptance Criteria
- Unique speaker count in reports matches `speakers_summary.csv`.
- Diarization yields realistic speaker count (no 900+ micro‑clusters).
- Re‑runs reuse preprocess/ASR caches; only changed phases re‑exec.
- Core pass on ~51‑min audio completes within ~10–15 minutes on the VM; enrichment dominates runtime when enabled.
