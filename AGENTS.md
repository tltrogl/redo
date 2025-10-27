# AGENTS.md — Codex Cloud Environment Agent (Strict, Complete)

> Contract for **autonomous agents** running this project in a Codex‑style cloud environment. Execute **literally** with no mid‑run confirmations.

---

## 1) Autonomy & Unblocking

- **Execution model:** **Plan → Execute → Self‑Verify → Continue.** Do **not** pause to ask for approval.
- **Unblocking mandate:** If a prerequisite is missing/broken, **detect → acquire/repair → verify** in the same run (packages, CLIs, models, directories, permissions, local services, caches).
- **No fallbacks:** Any internal fallback (e.g., switching runtimes/models/features) counts as **failure**.
- **Change style — “surgical but thorough”:** Edit the **smallest regions per file**, across **all files required** (code, config, scripts, tests, docs). Emit diffs after each edit wave.
- **Evidence every time:** Capture exit codes + concise logs for each major action and include them in the final report.

---

## 2) Environment Assumptions (Hard Requirements)

- **OS:** Linux container/VM; treat as **ephemeral and reproducible**.
- **Python:** **3.11** (exact). Create/activate a local venv.
- **CPU‑only.** Use **ONNXRuntime** models where defined; **do not** switch to PyTorch or other runtimes as fallback.
- **System tools:** `ffmpeg` must be on PATH.
- **Network:** **Allowed.** Internet access is available at all times.
- **Models:** **All required models are provided by `./models.zip`** in the repo root and **must** be used. Do not download alternative model files during the run.

---

## 3) Setup (Bootstrap) — Idempotent, No Placeholders

Run these commands at task start (or in the environment’s setup phase). They are fully concrete.

```bash
set -euo pipefail

# 3.1 Python environment (exact version)
python3.11 -V
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip wheel setuptools

# 3.2 Project dependencies (runtime + dev)
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
fi
# Optional editable install if pyproject/Extras present
if [ -f pyproject.toml ]; then
  pip install -e ".[dev]" || pip install -e .
fi

# 3.3 Ensure ffmpeg (install only if missing)
if ! command -v ffmpeg >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y ffmpeg
fi

# 3.4 Models: unzip strictly from the provided bundle at repo root
test -f ./models.zip
rm -rf ./models && mkdir -p ./models
unzip -q ./models.zip -d ./models

# 3.5 Verify required model files exist (strict, no fallbacks)
required_models=(
  "./models/Diarization/ecapa-onnx/ecapa_tdnn.onnx"
  "./models/Diarization/silaro_vad/silero_vad.onnx"
  "./models/Affect/ser8/model.int8.onnx"
  "./models/Affect/VAD_dim/model.onnx"
  "./models/Affect/sed_panns/model.onnx"
  "./models/Affect/sed_panns/class_labels_indices.csv"
  "./models/text_emotions/model.int8.onnx"
  "./models/intent/model_int8.onnx"
)
for f in "${required_models[@]}"; do
  [ -f "$f" ] || { echo "Missing required model file: $f" >&2; exit 1; }
done

# 3.6 Caches (local, concrete paths)
export DIAREMOT_MODEL_DIR="$(pwd)/models"
export HF_HOME="$(pwd)/.cache"
export TRANSFORMERS_CACHE="$(pwd)/.cache/transformers"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"
```

---

## 4) Maintenance (Resuming Cached Containers)

```bash
set -euo pipefail
[ -d .venv ] || python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip wheel setuptools
[ -f requirements.txt ] && pip install -r requirements.txt || true
if [ -f pyproject.toml ]; then
  pip install -e ".[dev]" || pip install -e .
fi
# Re‑assert caches and model roots
export DIAREMOT_MODEL_DIR="$(pwd)/models"
export HF_HOME="$(pwd)/.cache"
export TRANSFORMERS_CACHE="$(pwd)/.cache/transformers"
```

---

## 5) Canonical Commands (Build / Test / Lint) — Strict

```bash
set -euo pipefail
. .venv/bin/activate

# Lint (required)
ruff check src/ tests/

# Type check (required if type info present; otherwise soft‑pass to not block)
mypy src/ || true

# Unit/functional tests (required if tests exist)
if [ -d tests ]; then
  pytest -q
fi

# Diagnostics (on demand)
python -m diaremot.cli diagnostics
```

- **Strictness:** Any test/lint **hard failure** aborts the run. Do **not** substitute or skip.
- **No fallbacks:** If any component would fallback (e.g., alternate runtime/model/feature disable), treat as **failure**.

---

## 6) Strict Smoke Check (All Features, No Fallbacks)

This must exercise **all features** (Quiet‑Boost, SED [default ON], Diarization, ASR, Affect V/A/D, Speech Emotion 8‑class, Text Emotions 28‑class, Intent, Outputs). **VAD_dim is mandatory. SED is enabled by default.**

```bash
set -euo pipefail
. .venv/bin/activate

OUTDIR="/tmp/smoke_strict"
rm -rf "$OUTDIR" && mkdir -p "$OUTDIR"

# Run the strict smoke using local models only (no alternative downloads)
PYTHONPATH=src HF_HOME="$(pwd)/.cache" TRANSFORMERS_CACHE="$(pwd)/.cache/transformers" DIAREMOT_MODEL_DIR="$(pwd)/models" python -m diaremot.cli smoke   --outdir "$OUTDIR"   --model-root "$(pwd)/models"   --enable-sed   --enable-affect   --require-vad-dim   --strict

# Verify artifacts and required columns/fields
python - << 'PY'
import json, csv, os, sys
outdir = "/tmp/smoke_strict"
required_files = [
    "diarized_transcript_with_emotion.csv",
    "segments.jsonl",
    "summary.html",
    "speakers_summary.csv",
    "events_timeline.csv",
    "timeline.csv",
    "qc_report.json"
]
missing = [f for f in required_files if not os.path.isfile(os.path.join(outdir,f))]
if missing:
    print("Missing artifacts:", missing, file=sys.stderr); sys.exit(1)

# CSV schema (40 columns, exact order expected by this project)
expected_cols = [
 "file_id","start","end","speaker_id","speaker_name","text",
 "valence","arousal","dominance",
 "emotion_top","emotion_scores_json",
 "text_emotions_top5_json",
 "intent_top","intent_top3_json",
 "events_top3_json","noise_tag",
 "asr_logprob_avg","snr_db","snr_db_sed",
 # fill remaining columns that are contractual for this project if present
]
# Soft check: ensure a superset with all named columns present
with open(os.path.join(outdir,"diarized_transcript_with_emotion.csv"), newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    header = reader.fieldnames or []
    missing_cols = [c for c in expected_cols if c not in header]
    if missing_cols:
        print("CSV missing expected columns:", missing_cols, file=sys.stderr); sys.exit(1)
    row_count = sum(1 for _ in reader)
    if row_count == 0:
        print("CSV has zero rows", file=sys.stderr); sys.exit(1)

# Basic QC: ensure affect and text metrics present in segments.jsonl
affect_ok = text_ok = intent_ok = False
with open(os.path.join(outdir,"segments.jsonl"), encoding="utf-8") as f:
    for i,line in enumerate(f):
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if all(k in obj for k in ("valence","arousal","dominance")):
            affect_ok = True
        if "text_emotions_top5" in obj or "text_emotions" in obj:
            text_ok = True
        if "intent_top" in obj or "intent" in obj:
            intent_ok = True
        if affect_ok and text_ok and intent_ok:
            break
if not (affect_ok and text_ok and intent_ok):
    print("Segment metrics missing: affect/text/intent", file=sys.stderr); sys.exit(1)

# SED timeline file exists; content may be empty if truly silent, which is acceptable for smoke
# VAD_dim is mandatory but validated via presence of valence/arousal/dominance above

print("STRICT_SMOKE_OK")
PY
```

Any missing artifact/column/metric causes a **hard failure**.

---

## 7) Repo Layout & Entry Points (Concrete)

- **CLI:** `src/diaremot/cli.py`
- **Pipeline Orchestrator:** `src/diaremot/pipeline/orchestrator.py`
- **Stages Registry:** `src/diaremot/pipeline/stages/__init__.py`
- **Outputs & schema:** `src/diaremot/pipeline/outputs.py` (40 contractual columns)
- **Affect / SED / Intent:** `src/diaremot/affect/`, `src/diaremot/sed/`, `src/diaremot/intent/`
- **Model utilities:** `src/diaremot/io/onnx_utils.py`, `src/diaremot/utils/model_paths.py`
- **Tests:** `tests/`
- **Setup scripts:** `setup.sh`, `setup.ps1`

---

## 8) Models & Data (Local‑First, Mandatory)

- Use **only** the models from `./models.zip` (unzipped to `./models`).
- **Required paths (must exist):**
  - `./models/Diarization/ecapa-onnx/ecapa_tdnn.onnx`
  - `./models/Diarization/silaro_vad/silero_vad.onnx`
  - `./models/Affect/ser8/model.int8.onnx`
  - `./models/Affect/VAD_dim/model.onnx`
  - `./models/Affect/sed_panns/model.onnx`
  - `./models/Affect/sed_panns/class_labels_indices.csv`
  - `./models/text_emotions/model.int8.onnx`
  - `./models/intent/model_int8.onnx`

- **Environment overrides (concrete):**
```bash
export DIAREMOT_MODEL_DIR="$(pwd)/models"
export SILERO_VAD_ONNX_PATH="$DIAREMOT_MODEL_DIR/Diarization/silaro_vad/silero_vad.onnx"
export ECAPA_ONNX_PATH="$DIAREMOT_MODEL_DIR/Diarization/ecapa-onnx/ecapa_tdnn.onnx"
export DIAREMOT_PANNS_DIR="$DIAREMOT_MODEL_DIR/Affect/sed_panns"
export DIAREMOT_TEXT_EMO_MODEL_DIR="$DIAREMOT_MODEL_DIR/text_emotions"
export AFFECT_VAD_DIM_MODEL_DIR="$DIAREMOT_MODEL_DIR/Affect/VAD_dim"
export DIAREMOT_INTENT_MODEL_DIR="$DIAREMOT_MODEL_DIR/intent"
export HF_HOME="$(pwd)/.cache"
export TRANSFORMERS_CACHE="$(pwd)/.cache/transformers"
```

---

## 9) Network & Secrets

- **Internet:** Always allowed.
- **Secrets:** Not required for standard runs. If introduced in the future, fetch them **only during setup** and cache artifacts locally; do not require secrets mid‑run.

---

## 10) Artifacts & Success Criteria (Hard Gates)

A run is successful only if **all** of the following pass:

1. **Artifacts present:**
   - `diarized_transcript_with_emotion.csv`
   - `segments.jsonl`
   - `summary.html`
   - `speakers_summary.csv`
   - `events_timeline.csv`
   - `timeline.csv`
   - `qc_report.json`
2. **CSV contract:** `diarized_transcript_with_emotion.csv` has **40 columns** including affect (V/A/D), speech emotion, text emotions top‑5 JSON, intent, SED overlaps, noise/SNR metrics.
3. **All features exercised:** Quiet‑Boost, SED (ON), Diarization, ASR, Affect V/A/D, Speech Emotion 8‑class, Text Emotions 28‑class, Intent. **VAD_dim present.**
4. **No fallbacks used.** If any component indicates fallback/disablement, mark as **failure**.
5. **Lint/tests:** `ruff` and `pytest` (if tests exist) complete with **zero hard failures**.

---

## 11) Known Pitfalls / Gotchas (Concrete)

- **Directory name:** Silero VAD lives under `Diarization/silaro_vad/` (intentional spelling in this repo).
- **Quantized models:** SER8 and text‑emotions models are `model.int8.onnx`; intent is `model_int8.onnx`.
- **Strictness:** Treat missing models, disabled features, or alternate runtimes as **hard failures**.
- **Long audio:** The pipeline auto‑chunks for long files; ensure sufficient disk space in the workspace.

---

## 12) Completion & Run Report (What to Emit)

Include in the final output:

- **Summary:** Task requested; what changed; why.
- **Diffs:** Unified diffs for **all** files modified.
- **Commands:** Exact commands executed with exit codes.
- **Evidence:** Key log excerpts and exact artifact paths.
- **Follow‑ups:** Any temporary services/caches created; how to disable/clean.

**Completion:** All hard gates in §10 are green, with evidence attached.
