# DiaRemot TODO — Operational Checklist (v2.2.0)

Purpose: a quick, editable checklist to drive the pipeline to a stable, fast state on local and VM environments. Mirrors AGENTS.md §11.

## Quick Checklist
- [ ] VAD ONNX is loaded (no Torch fallback): envs set; diagnostic confirms session.
- [ ] Diarization yields realistic unique speakers (e.g., 2–10; not hundreds).
- [ ] Reports show unique speaker count and correct dominance.
- [ ] Preprocess/ASR caches are reused; only necessary stages rerun.
- [ ] VM perf env in place; ECAPA batches increased; ASR threads set.
- [ ] Two‑phase flow available; enrichment can run independently.

---

## 1) Silero VAD ONNX Loading
- Why it matters: If VAD doesn’t load (or falls back), you get unstable speech regions → over‑segmentation.
- Do (VM/local):
  ```bash
  export SILERO_VAD_TORCH=0
  export SILERO_VAD_ONNX_PATH=/home/<user>/dia/diaremot2-on/models/diarization/silaro_vad/silero_vad.onnx
  ```
  Replace `<user>` as appropriate.
- Verify:
  ```bash
  python - << 'PY'
  from diaremot.pipeline.diarization.vad import SileroVAD
  v=SileroVAD(0.22,0.10,backend='onnx'); print('onnx_loaded=', v.session is not None)
  PY
  ```
- Rollback: Unset the var or point to a different ONNX model path if you have multiple installs.

## 2) Diarization Stabilization (VAD + Clustering Bounds)
- Why it matters: Liberal VAD + unbounded AHC → hundreds of micro‑clusters.
- Do:
  - Flags: `--vad-backend onnx --vad-threshold 0.22 --vad-min-speech-sec 0.25 --vad-min-silence-sec 0.25 --vad-speech-pad-sec 0.20`
  - `--speaker_limit 4` to bound clusters (tune 3–6 based on your data).
  - Keep: `ahc_distance_threshold 0.12–0.16`, `min_turn_sec ≥ 1.2`, `max_gap_to_merge_sec ≈ 1.0`, post‑merge distance ≥ 0.30.
- Verify:
  ```bash
  python - << 'PY'
  import csv,collections
  d=collections.defaultdict(float)
  for r in csv.DictReader(open('audio/outs/<outdir>/speakers_summary.csv',encoding='utf-8')):
      d[(r.get('speaker_name') or r.get('speaker_id') or '').strip()]+=float(r.get('total_duration',0) or 0)
  print('unique_speakers=',len(d))
  print('top5=',sorted(d.items(),key=lambda kv:kv[1],reverse=True)[:5])
  PY
  ```
- Rollback: Relax limits (e.g., `--speaker_limit 6`) if you under‑split; adjust VAD threshold ±0.02.

## 3) Preprocess Cache Crash (Guard NameError)
- Why it matters: A NameError in `_load_diar_tx_caches` aborts runs mid‑preprocess.
- Do (code fix): inside `src/diaremot/pipeline/stages/preprocess.py`, ensure the helper logs via `pipeline.corelog.stage(...)` instead of `guard.progress(...)`, or pass `guard` into the helper.
- Verify: rerun a short file; confirm no crash and cache is written (log shows `saved cache`).
- Rollback: `git checkout -- src/diaremot/pipeline/stages/preprocess.py` to restore.

## 4) Report Correctness (Speakers + Dominance)
- Why it matters: Reports showed 1000+ speakers due to counting rows/estimates.
- Do (code fix):
  - `src/diaremot/summaries/narrative_builder.py`: count unique speaker labels; compute dominance from summed per‑speaker `total_duration`.
  - `src/diaremot/summaries/html_summary_generator.py` and `pdf_summary_generator.py`: show unique count; compute leader share vs total speech, not full audio.
- Verify: regenerate `summary.html/pdf`; “Speakers” matches `speakers_summary.csv` uniques, and dominance looks plausible.
- Rollback: `git checkout -- src/diaremot/summaries/*`.

## 5) VM Performance Guardrails
- Why it matters: Hidden BLAS threads waste CPU; ECAPA under‑batches; ASR silently single‑threads.
- Do:
  ```bash
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
  export DIAREMOT_ECAPA_MAX_BATCH=1024
  # choose ASR threads based on cores (4–8 is typical)
  ```
- Verify: diarization step wall‑time decreases; CPU stays saturated without thrash; RAM stays within limits.
- Rollback: unset envs; reduce ECAPA batch if RAM spikes.

## 6) Two‑Phase Flow (Core → Enrichment)
- Why it matters: Faster iteration and fault isolation.
- Do:
  - Phase A (core only): `--no-enable-sed --disable-affect`.
  - Phase B (enrichment): rerun without `--disable-affect` and with `--enable-sed`.
- Verify: Phase B reuses Phase A artifacts; reports update without re‑doing diarization/ASR.
- Rollback: merge back to single command if desired.

## 7) CLI Ergonomics
- Why it matters: Avoid code edits for clustering switches.
- Do: ensure CLI exposes `--speaker_limit`, `--min-speakers`, `--max-speakers`, `--clustering-backend {ahc,spectral}`.
- Verify: `python -m diaremot.pipeline.cli_entry -h` shows the flags.

## 8) Standardize Preprocess Defaults
- Why it matters: Cache signature drift triggers reprocessing.
- Do: settle on noise reduction ON, loudness mode = `asr`. Document expected flags in AGENTS.md.
- Verify: subsequent runs log `loaded cached preprocessed audio`.

## 9) Sanity Checks (QC / Logs)
- Why it matters: Early detection of regressions.
- Do:
  - Warn if `speakers_est` ≫ unique speakers (e.g., est>50 and unique<10).
  - Validate the CSV schema has 53 columns.
  - If Silero ONNX missing, fail‑soft and print `SILERO_VAD_ONNX_PATH` hint.

---

## Snippets
- Targeted cache clear by audio SHA (keeps models):
  ```bash
  python - << 'PY'
  import hashlib, pathlib, shutil
  a=pathlib.Path('audio/yourfile.ext'); h=hashlib.blake2s(digest_size=16)
  with a.open('rb') as f:
      for chunk in iter(lambda:f.read(1024*1024), b''):
          h.update(chunk)
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

## Acceptance Criteria
- [ ] Unique speaker count in reports == unique speakers in `speakers_summary.csv`.
- [ ] Realistic speaker count (no 900+ micro‑clusters).
- [ ] Re‑runs reuse preprocess/ASR caches; only changed phases re‑exec.
- [ ] Core pass on ~51‑min audio completes within ~10–15 minutes on the VM; enrichment dominates runtime when enabled.
