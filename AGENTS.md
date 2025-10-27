````markdown
# AGENTS.md — Codex Cloud Environment Agent (Strict, Complete)

> Contract for **autonomous agents** running this project in a Codex-style cloud environment. Execute **literally** with no mid-run confirmations.

---

## 1) Autonomy & Unblocking

- **Execution model:** **Plan → Execute → Self-Verify → Continue.** Do **not** pause to ask for approval.
- **Unblocking mandate:** If a prerequisite is missing/broken, **detect → acquire/repair → verify** in the same run (packages, CLIs, models, directories, permissions, local services, caches).
- **No fallbacks:** Any internal fallback (changing runtimes/models/features, or skipping a required stage) is a **failure**.
- **Change style — “surgical but thorough”:** Edit the **smallest regions per file**, across **all files required** (code, config, scripts, tests, **docs**). Emit diffs after each edit wave.
- **Docs always in sync:** If behavior, CLI flags, config, paths, outputs/schema, or runtime assumptions change, update **README.md**, **AGENTS.md**, and any **docs/** in the **same run** and ensure docs build clean (if a builder exists).
- **Evidence on every iteration:** Capture exit codes + concise logs for each major action and include them in the final report.

---

## 2) Environment Assumptions (Hard Requirements)

- **OS:** Linux container/VM; treat as **ephemeral and reproducible**.
- **Python:** **3.11** (exact). Create/activate a local venv.
- **CPU-only.** Use **ONNXRuntime** models where defined; **do not** switch to other runtimes.
- **System tools:** `ffmpeg` must be on PATH.
- **Network:** **Allowed** at all times.
- **Models:** Must be available **locally** under `./models` (see §3.4 and §8). Use the official release bundle if local assets are not already present.

---

## 3) Setup (Bootstrap) — Idempotent, No Placeholders

Run these commands at task start (or in the environment’s setup phase).

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
if [ -f pyproject.toml ]; then
  pip install -e ".[dev]" || pip install -e .
fi

# 3.3 Ensure ffmpeg (install only if missing)
if ! command -v ffmpeg >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y ffmpeg
fi

# 3.4 Models: prefer existing local ./models; otherwise unzip from models.zip; otherwise fetch official bundle
exists_any() { for p in "$@"; do [ -f "$p" ] && return 0; done; return 1; }

have_all_models() {
  # Core ONNX assets (accept common alias layouts)
  exists_any "./models/Diarization/ecapa-onnx/ecapa_tdnn.onnx" "./models/ecapa_onnx/ecapa_tdnn.onnx" "./models/ECAPA_ONNX/ecapa_tdnn.onnx" "./models/ecapa_tdnn.onnx" || return 1
  exists_any "./models/Diarization/silaro_vad/silero_vad.onnx" "./models/silero_vad.onnx" "./models/SILERO_VAD.onnx" || return 1
  exists_any "./models/Affect/ser8/model.int8.onnx" "./models/ser8-onnx/model.int8.onnx" "./models/SER8-ONNX/model.int8.onnx" || return 1
  exists_any "./models/Affect/VAD_dim/model.onnx" "./models/VAD_dim/model.onnx" "./models/vad_dim/model.onnx" || return 1
  exists_any "./models/Affect/sed_panns/model.onnx" "./models/panns/model.onnx" "./models/PANNS/model.onnx" || return 1
  exists_any "./models/Affect/sed_panns/class_labels_indices.csv" "./models/panns/class_labels_indices.csv" "./models/PANNS/class_labels_indices.csv" || return 1
  exists_any "./models/text_emotions/model.int8.onnx" "./models/goemotions-onnx/model.int8.onnx" "./models/GoEmotions-ONNX/model.int8.onnx" || return 1
  exists_any "./models/intent/model_int8.onnx" "./models/bart/model_int8.onnx" "./models/BART/model_int8.onnx" || return 1
  # Faster-Whisper (CTranslate2) tiny.en — require local files; do NOT download at run time
  if ! ( \
    [ -f "./models/asr/whisper-ct2-tiny-en/model.bin" ] && [ -f "./models/asr/whisper-ct2-tiny-en/tokenizer.json" ] \
    || [ -f "./models/whisper-ct2/tiny.en/model.bin" ] && [ -f "./models/whisper-ct2/tiny.en/tokenizer.json" ] \
    || [ -f "./models/faster-whisper/tiny.en/model.bin" ] && [ -f "./models/faster-whisper/tiny.en/tokenizer.json" ] \
    || [ -f "./models/ctranslate2/whisper-tiny.en/model.bin" ] && [ -f "./models/ctranslate2/whisper-tiny.en/tokenizer.json" ] \
  ); then return 1; fi
  return 0
}

if have_all_models; then
  echo "Using existing local models under ./models"
else
  if [ -f models.zip ]; then
    echo "models.zip found locally; verifying checksum and extracting..."
    echo "3cc2115f4ef7cd4f9e43cfcec376bf56ea2a8213cb760ab17b27edbc2cac206c  models.zip" | sha256sum -c -
  else
    echo "Fetching official models bundle..."
    curl -L "https://github.com/tltrogl/diaremot2-ai/releases/download/v2.AI/models.zip" -o models.zip
    echo "3cc2115f4ef7cd4f9e43cfcec376bf56ea2a8213cb760ab17b27edbc2cac206c  models.zip" | sha256sum -c -
  fi
  rm -rf ./models && mkdir -p ./models
  unzip -q models.zip -d ./models
  have_all_models || { echo "Models missing after extraction. Check bundle and paths." >&2; exit 1; }
fi

# 3.5 Caches (local, concrete paths)
export DIAREMOT_MODEL_DIR="$(pwd)/models"
export HF_HOME="$(pwd)/.cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$HF_HOME/torch"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_MAX_THREADS=4
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME"
````

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
# Re-assert caches and model roots
export DIAREMOT_MODEL_DIR="$(pwd)/models"
export HF_HOME="$(pwd)/.cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$HF_HOME/torch"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_MAX_THREADS=4
```

---

## 5) Canonical Commands (Build / Test / Lint) — Strict

```bash
set -euo pipefail
. .venv/bin/activate

# Lint (required)
ruff check src/ tests/

# Type check (best-effort if no stubs/hints)
mypy src/ || true

# Unit/functional tests (required if tests exist)
if [ -d tests ]; then
  pytest -q
fi

# Diagnostics (on demand)
python -m diaremot.cli diagnostics

# Documentation build (only if Sphinx config exists)
if [ -f docs/conf.py ]; then
  sphinx-build -b html docs site -W
fi
```

### 5.1 Lint Profiles — Standard vs Strict

**Standard lint (dev/WIP):** fast feedback while coding; core hygiene only.
**Strict lint (CI/agent gate):** production quality; blocks on imports/order, bugbear/complexity, py-upgrades, security, **docstrings**.

**Agent rule:** the **agent always runs strict**. Developers may use standard locally if desired.

**Ruff recommendations (pyproject.toml):**

```toml
[tool.ruff]
target-version = "py311"
line-length = 100

# Standard (dev): core rules only
[tool.ruff.lint]
select = ["E","F","I","UP","B","C4","RUF"]
ignore = ["D","W","ANN","S"]
```

Optional separate strict config for CI/agent:

```toml
# pyproject.strict.toml
[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["ALL"]
ignore = ["D203","D212","COM812","COM819","ISC001"]
```

**Agent invocation (strict):**

```bash
ruff check --config pyproject.strict.toml src/ tests/
```

* **Strictness:** Any lint/test/docs **hard failure** aborts the run. Do **not** substitute or skip.
* **No fallbacks:** If any component would fallback (alternate runtime/model/feature disable), treat as **failure**.

---

## 6) Strict Smoke Check (All Features, No Fallbacks)

Must exercise **all features**: Quiet-Boost, **SED (ON by default)**, Diarization, ASR, Affect **V/A/D**, Speech Emotion (8-class), Text Emotions (28-class), Intent, and outputs. **VAD_dim is mandatory.**

```bash
set -euo pipefail
. .venv/bin/activate

OUTDIR="/tmp/smoke_strict"
rm -rf "$OUTDIR" && mkdir -p "$OUTDIR"

# Strict smoke using local models only (no alternative downloads)
PYTHONPATH=src HF_HOME="$(pwd)/.cache" TRANSFORMERS_CACHE="$(pwd)/.cache/transformers" DIAREMOT_MODEL_DIR="$(pwd)/models" \
python -m diaremot.cli smoke \
  --outdir "$OUTDIR" \
  --model-root "$(pwd)/models" \
  --enable-sed \
  --enable-affect \
  --require-vad-dim \
  --strict

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

# Required CSV columns (contract subset checked here)
expected_cols = [
 "file_id","start","end","speaker_id","speaker_name","text",
 "valence","arousal","dominance",
 "emotion_top","emotion_scores_json",
 "text_emotions_top5_json",
 "intent_top","intent_top3_json",
 "events_top3_json","noise_tag",
 "asr_logprob_avg","snr_db","snr_db_sed"
]
with open(os.path.join(outdir,"diarized_transcript_with_emotion.csv"), newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    header = reader.fieldnames or []
    missing_cols = [c for c in expected_cols if c not in header]
    if missing_cols:
        print("CSV missing expected columns:", missing_cols, file=sys.stderr); sys.exit(1)
    if sum(1 for _ in reader) == 0:
        print("CSV has zero rows", file=sys.stderr); sys.exit(1)

# Segment-level checks for affect/text/intent presence
affect_ok = text_ok = intent_ok = False
with open(os.path.join(outdir,"segments.jsonl"), encoding="utf-8") as f:
    for line in f:
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

print("STRICT_SMOKE_OK")
PY
```

Any missing artifact/column/metric is a **hard failure**.

---

## 7) Repo Layout & Entry Points (Concrete)

* **CLI:** `src/diaremot/cli.py`
* **Pipeline Orchestrator:** `src/diaremot/pipeline/orchestrator.py`
* **Stages Registry:** `src/diaremot/pipeline/stages/__init__.py`
* **Outputs & schema:** `src/diaremot/pipeline/outputs.py` (CSV columns listed above)
* **Affect / SED / Intent:** `src/diaremot/affect/`, `src/diaremot/sed/`, `src/diaremot/intent/`
* **Model utilities:** `src/diaremot/io/onnx_utils.py`, `src/diaremot/utils/model_paths.py`
* **Tests:** `tests/`
* **Setup scripts:** `setup.sh`, `setup.ps1`

---

## 8) Models & Data (Release Asset or Local, Mandatory)

* **Source (official bundle):**
  `https://github.com/tltrogl/diaremot2-ai/releases/download/v2.AI/models.zip`
  **SHA256:** `3cc2115f4ef7cd4f9e43cfcec376bf56ea2a8213cb760ab17b27edbc2cac206c`
* **Location:** `./models/` (unzipped). Alias-aware directory names supported (e.g., `ser8-onnx/`, `goemotions-onnx/`, `bart/`, `panns/`, `ecapa_onnx/`, or root `silero_vad.onnx`), but the **required files** must exist.
* **CT2 ASR:** **Faster-Whisper (CTranslate2) tiny.en** must be present locally (e.g., `models/asr/whisper-ct2-tiny-en/{model.bin,tokenizer.json}` or equivalent alias).
* **Environment (concrete):**

```bash
export DIAREMOT_MODEL_DIR="$(pwd)/models"
export SILERO_VAD_ONNX_PATH="$DIAREMOT_MODEL_DIR/Diarization/silaro_vad/silero_vad.onnx"
export ECAPA_ONNX_PATH="$DIAREMOT_MODEL_DIR/Diarization/ecapa-onnx/ecapa_tdnn.onnx"
export DIAREMOT_PANNS_DIR="$DIAREMOT_MODEL_DIR/Affect/sed_panns"
export DIAREMOT_TEXT_EMO_MODEL_DIR="$DIAREMOT_MODEL_DIR/text_emotions"
export AFFECT_VAD_DIM_MODEL_DIR="$DIAREMOT_MODEL_DIR/Affect/VAD_dim"
export DIAREMOT_INTENT_MODEL_DIR="$DIAREMOT_MODEL_DIR/intent"
export HF_HOME="$(pwd)/.cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$HF_HOME/torch"
```

---

## 9) Change Styles Policy (Defaults)

* **Default:** **Surgical but thorough** (small diffs; complete system coverage including tests/smoke/**docs**).
* **For modularizing transcription (and similar subsystem work):**
  **Branch-by-abstraction** (introduce a stable interface), plus **expand/contract** for call sites. Switch over when green; remove old path.
* **Boy-Scout (adjacent-only) — allowed when relevant:**
  Minor, nearby cleanups that **directly** improve clarity of the touched code (e.g., fix an obviously misleading name, tighten a docstring). Keep it **minimal and adjacent**; no drive-by refactors.

---

## 10) Artifacts & Success Criteria (Hard Gates)

A run is successful only if **all** pass:

1. **Artifacts present:**

   * `diarized_transcript_with_emotion.csv`
   * `segments.jsonl`
   * `summary.html`
   * `speakers_summary.csv`
   * `events_timeline.csv`
   * `timeline.csv`
   * `qc_report.json`
2. **CSV contract:** `diarized_transcript_with_emotion.csv` contains all required columns listed in §6.
3. **All features exercised:** Quiet-Boost, **SED (ON)**, Diarization, ASR, Affect **V/A/D**, Speech Emotion 8-class, Text Emotions 28-class, Intent. **VAD_dim present.**
4. **No fallbacks used.**
5. **Lint/tests:** `ruff` and `pytest` (if tests exist) complete with **zero hard failures**.
6. **Documentation updated & builds clean:** README/AGENTS/docs reflect the change set.

   * If **Sphinx** is present (`docs/conf.py`), `sphinx-build -W` must succeed.
   * If no doc builder is present, the **Run Report** must list updated files and show diffs.

---

## 11) Completion & Run Report (What to Emit)

Include in final output:

* **Summary:** Task requested; what changed; why.
* **Diffs:** Unified diffs for **all** files modified.
* **Commands:** Exact commands executed with exit codes.
* **Evidence:** Key log excerpts and exact artifact paths.
* **Docs delta:** List which docs were updated (README.md, AGENTS.md, docs/**) and why.
* **Follow-ups:** Any temporary services/caches created; how to disable/clean.

**Completion:** All hard gates in §10 are green, with evidence attached.

---

## Appendix A — Codex Environment Setup Guide

This document captures the exact steps used to bootstrap the DiaRemot Codex worker environment for CPU-only smoke testing. Follow this sequence to reproduce the runtime that validated the latest smoketest run.

### 1. System prerequisites

Start from **Ubuntu 22.04 LTS** (the Codex worker baseline).
Update the package index and install the multimedia libraries required by the pipeline:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libsm6 libxext6
```

* `ffmpeg` enables audio resampling, loudness normalisation, and segment extraction.
* `libsm6` and `libxext6` satisfy OpenCV and Pillow runtime dependencies.

### 2. Python toolchain

Ensure Python **3.11** is available. Install from deadsnakes if the base image does not already include it:

```bash
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
```

Create the project virtual environment at the repo root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Upgrade the packaging toolchain inside the venv:

```bash
python -m pip install --upgrade pip wheel setuptools
```

### 3. Install DiaRemot dependencies

Install the pinned runtime stack:

```bash
pip install -r requirements.txt
```

Install DiaRemot in editable mode for local development and CLI access:

```bash
pip install -e .
```

### 4. Hydrate model assets

Download and validate the curated `models.zip` bundle used by Codex workers (or place your already-downloaded `models.zip` in the repo root first to skip network fetch):

```bash
curl -L https://github.com/tltrogl/diaremot2-ai/releases/download/v2.AI/models.zip -o models.zip
echo "3cc2115f4ef7cd4f9e43cfcec376bf56ea2a8213cb760ab17b27edbc2cac206c  models.zip" | sha256sum -c -
unzip -q models.zip -d ./models
```

The archive expands into an **alias-aware** directory layout consumed by the pipeline (e.g., `ser8-onnx/`, `goemotions-onnx/`, `bart/`, `panns/`, `ecapa_onnx/`, or a root `silero_vad.onnx`). Case-folded lookups mean variations such as `GoEmotions-ONNX/` also resolve correctly. **Preserve the extracted directory structure; do not move ONNX files to the repository root.** Ensure the **Faster-Whisper (CTranslate2) tiny.en** directory (with `model.bin` + `tokenizer.json`) is present locally under one of the accepted aliases (see §3.4).

### 5. Configure caches and environment variables

Add the following exports to your shell profile (or set them in CI) to mirror Codex defaults:

```bash
export DIAREMOT_MODEL_DIR="$(pwd)/models"
export HF_HOME="$(pwd)/.cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$HF_HOME/torch"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_MAX_THREADS=4
```

These settings enforce **local-first model resolution** and consistent **CPU threading** across runs.

### 6. Validate the environment

Prime the caches by running the diagnostics and smoke commands:

```bash
python -m diaremot.cli diagnostics

PYTHONPATH=src HF_HOME=.cache \
python -m diaremot.cli smoke \
  --outdir /tmp/smoke_test \
  --model-root ./models \
  --enable-affect
```

Confirm all **11 pipeline stages** report `PASS` in the smoke summary and that `/tmp/smoke_test/` contains the CSV/JSON/HTML artefacts.

### 7. Housekeeping

Keep `.cache/` and `./models` persisted between runs to avoid repeated downloads. Regenerate the environment by re-running the steps above after dependency updates or release upgrades.

```

::contentReference[oaicite:0]{index=0}
```
