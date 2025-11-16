# AGENTS.md — Local Codex IDE Agent Handbook

This document describes how the interactive Codex agent collaborates with the user when working on DiaRemot from a local IDE session. Everything below assumes the project is already installed and working; the primary job is to reason about changes, verify outcomes, and communicate clearly.

---

## 1. Role Overview

- **Plan first.** Outline intended edits, tests, or inspections and wait for explicit approval before touching the repo or running commands.
- **Execute surgically.** Apply only the approved changes, keeping edits as small and focused as possible.
- **Observe, don’t assume.** Use the existing artefacts (`logs/`, `outputs/`, `qc_report.json`, etc.) to understand pipeline behaviour before altering configuration.
- **Report concisely.** Summarise what changed, what you ran, and what you observed. Note any follow-up actions for the user.

---

## 2) Environment Assumptions (Hard Requirements)

- **OS:** Linux container/VM; treat as **ephemeral and reproducible**.
- **Python:** **3.11** (exact). Create/activate a local venv.
- **CPU‑only.** Use **ONNXRuntime** models where defined; **do not** switch to PyTorch or other runtimes as fallback.
- **System tools:** `ffmpeg` must be on PATH.
- **Network:** **Allowed.** Internet access is available at all times.
- **Models:** **All required models are provided by `./assets/models.zip`** (SHA-256 `33d0a9194de3cbd667fd329ab7c6ce832f4e1373ba0b1844ce0040a191abc483`). Extract this archive locally on first run and reuse the unpacked assets; do **not** download alternative model files during the run.
## 2. Environment Assumptions

- Python 3.11, the virtual environment (`.venv`), ffmpeg, and dependencies are already in place.
- The primary model bundle lives at `D:/models`; environment variable `DIAREMOT_MODEL_DIR` should already point there. Only fall back to `./assets/models.zip` if instructed.
- Hugging Face / CT2 caches exist under `.cache/`; leave them intact unless you have a reason to rebuild.
- Network access is available for additional downloads, but unnecessary in the normal workflow.

*If any prerequisite is missing, flag it in your plan before making the change.*

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

# 3.4 Models: download (idempotent) and unzip strictly from the provided bundle
mkdir -p ./assets
if [ ! -f ./assets/models.zip ]; then
  curl -sSLf https://github.com/tltrogl/diaremot2-ai/releases/download/v2.AI/models.zip -o ./assets/models.zip
fi
sha256sum ./assets/models.zip | grep -qi "33d0a9194de3cbd667fd329ab7c6ce832f4e1373ba0b1844ce0040a191abc483"
rm -rf ./models
unzip -q ./assets/models.zip -d ./models

# 3.5 Verify required model files exist (strict, no fallbacks)
required_models=(
  "./models/Diarization/ecapa-onnx/ecapa_tdnn.onnx"
  "./models/Diarization/silero_vad/silero_vad.onnx"
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
## 3. Standard Workflow

1. **Gather context.** Review the repo state (`git status`, recent logs, pending outputs) to understand current conditions.
2. **Propose a plan.** List the smallest set of steps needed (e.g., “inspect file X,” “modify Y,” “run smoke test”). Wait for approval.
3. **Execute approved steps.** Activate the existing venv if needed (`source .venv/bin/activate` or `.\.venv\Scripts\activate`). Run only the commands that were agreed upon.
4. **Inspect diagnostics.** For pipeline runs, check `logs/run.jsonl`, `outputs/qc_report.json`, and generated artefacts. Look for fallbacks, missing columns, or dependency warnings.
5. **Communicate results.** Provide a brief summary, list of commands with exit codes, notable log excerpts, and any follow-up recommendations.

---

## 4. Useful Commands (Run Only With Approval)

```bash
set -euo pipefail
[ -d .venv ] || python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip wheel setuptools
[ -f requirements.txt ] && pip install -r requirements.txt || true
if [ -f pyproject.toml ]; then
  pip install -e ".[dev]" || pip install -e .
fi
# Reassert caches and reuse the unpacked model bundle when available
export DIAREMOT_MODEL_DIR="$(pwd)/models"
if [ ! -f "$DIAREMOT_MODEL_DIR/Diarization/ecapa-onnx/ecapa_tdnn.onnx" ]; then
  mkdir -p assets "$DIAREMOT_MODEL_DIR"
  if [ ! -f assets/models.zip ]; then
    curl -sSLf https://github.com/tltrogl/diaremot2-ai/releases/download/v2.AI/models.zip -o assets/models.zip
  fi
  sha256sum assets/models.zip | grep -qi "33d0a9194de3cbd667fd329ab7c6ce832f4e1373ba0b1844ce0040a191abc483"
  unzip -qn assets/models.zip -d "$DIAREMOT_MODEL_DIR"
fi
export HF_HOME="$(pwd)/.cache"
export TRANSFORMERS_CACHE="$(pwd)/.cache/transformers"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"
```

---

## 5) Canonical Commands (Build / Test / Lint) — Strict

```bash
set -euo pipefail
. .venv/bin/activate

# Lint (required)
# Pipeline execution on real audio (outputs → <input dir>/outs/<stem> by default)
python -m diaremot.cli run --input input.wav \
  --model-root "${DIAREMOT_MODEL_DIR:-./models}" --enable-sed --enable-affect
# Or use the helper wrapper (auto-detects threads on the VM)
bash scripts/diaremot_run.sh run input.wav

# Synthetic smoke test (generates demo audio)
python -m diaremot.cli smoke --outdir /tmp/smoke --enable-affect \
  --model-root "${DIAREMOT_MODEL_DIR:-./models}"

# Resume from checkpoints (reuses the same derived outdir unless overridden)
python -m diaremot.cli resume --input input.wav

# Lint / tests
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

## 6) Strict Smoke Check (All Features, Zero Fallbacks)

Execute the CPU smoke with every optional feature enabled. The manifest that the CLI prints must be captured and audited: **any fallback, placeholder, or disabled feature is a hard failure**.

```bash
set -euo pipefail
. .venv/bin/activate

OUTDIR="/tmp/smoke_strict"
rm -rf "$OUTDIR" && mkdir -p "$OUTDIR"

PYTHONPATH=src \
HF_HOME="$(pwd)/.cache" \
TRANSFORMERS_CACHE="$(pwd)/.cache/transformers" \
DIAREMOT_MODEL_DIR="$(pwd)/models" \
python -m diaremot.cli smoke \
  --outdir "$OUTDIR" \
  --model-root "$(pwd)/models" \
  --enable-affect \
  | tee "$OUTDIR/manifest.json"

python3.11 - <<'PY'
import csv, json, os, sys, pathlib
outdir = pathlib.Path("/tmp/smoke_strict")
manifest_path = outdir / "manifest.json"
try:
    manifest = json.load(open(manifest_path, encoding="utf-8"))
except Exception as exc:
    print(f"Manifest missing or invalid: {exc}", file=sys.stderr)
    sys.exit(1)

required_keys = {
    "csv": "diarized_transcript_with_emotion.csv",
    "jsonl": "segments.jsonl",
    "timeline": "timeline.csv",
    "summary_html": "summary.html",
    "qc_report": "qc_report.json",
}

for key, default_name in required_keys.items():
    path = pathlib.Path(manifest["outputs"].get(key, outdir / default_name))
    if not path.exists():
        print(f"Missing required artifact for {key}: {path}", file=sys.stderr)
        sys.exit(1)

csv_path = pathlib.Path(manifest["outputs"]["csv"])
with open(csv_path, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    header = reader.fieldnames or []

EXPECTED_COLUMNS = [
    "file_id","start","end","speaker_id","speaker_name","text",
    "valence","arousal","dominance","emotion_top","emotion_scores_json",
    "text_emotions_top5_json","text_emotions_full_json",
    "intent_top","intent_top3_json","events_top3_json","noise_tag",
    "asr_logprob_avg","snr_db","snr_db_sed","wpm","duration_s","words",
    "pause_ratio","low_confidence_ser","vad_unstable","affect_hint",
    "pause_count","pause_time_s","f0_mean_hz","f0_std_hz","loudness_rms",
    "disfluency_count","error_flags","vq_jitter_pct","vq_shimmer_db",
    "vq_hnr_db","vq_cpps_db","voice_quality_hint"
]

missing_cols = [col for col in EXPECTED_COLUMNS if col not in header]
if missing_cols or len(header) != len(EXPECTED_COLUMNS):
    print(f"CSV schema mismatch. Missing {missing_cols}; column_count={len(header)}", file=sys.stderr)
    sys.exit(1)

rows = list(reader)
if not rows:
    print("CSV has zero rows", file=sys.stderr)
    sys.exit(1)

segments_jsonl = pathlib.Path(manifest["outputs"]["jsonl"])
with open(segments_jsonl, encoding="utf-8") as handle:
    affect_ok = text_ok = intent_ok = False
    for line in handle:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        affect_ok = affect_ok or all(k in obj for k in ("valence","arousal","dominance"))
        text_ok = text_ok or bool(obj.get("text_emotions_top5") or obj.get("text_emotions"))
        intent_ok = intent_ok or bool(obj.get("intent_top") or obj.get("intent"))
        if affect_ok and text_ok and intent_ok:
            break
if not (affect_ok and text_ok and intent_ok):
    print("Segment metrics missing affect/text/intent payloads", file=sys.stderr)
    sys.exit(1)

issues = manifest.get("issues") or []
if issues:
    print(f"Manifest issues detected (treat as fallback): {issues}", file=sys.stderr)
    sys.exit(1)
if not manifest.get("dependency_ok", True):
    print("Dependency summary reported unhealthy status", file=sys.stderr)
    sys.exit(1)

qc_path = pathlib.Path(manifest["outputs"]["qc_report"])
qc = json.load(open(qc_path, encoding="utf-8"))
warnings = qc.get("warnings") or []
failure_markers = ("fallback", "placeholder", "skipped")
flagged = [w for w in warnings if any(marker in w.lower() for marker in failure_markers)]
if flagged:
    print(f"Fallback indicators discovered in QC warnings: {flagged}", file=sys.stderr)
    sys.exit(1)

log_path = pathlib.Path("logs/run.jsonl")
if log_path.exists():
    for line in log_path.read_text(encoding="utf-8").splitlines():
        lower = line.lower()
        if any(marker in lower for marker in failure_markers):
            print(f"Fallback detected in logs: {line}", file=sys.stderr)
            sys.exit(1)
PY
```

Any missing artifact, schema mismatch, or fallback indicator causes a **hard failure**.

---

## 7) Repo Layout & Entry Points (Concrete)

- **CLI:** `src/diaremot/cli.py`
- **Pipeline Orchestrator:** `src/diaremot/pipeline/orchestrator.py`
- **Stages Registry:** `src/diaremot/pipeline/stages/__init__.py`
- **Outputs & schema:** `src/diaremot/pipeline/outputs.py` (53 contractual columns)
- **Affect / SED / Intent:** `src/diaremot/affect/`, `src/diaremot/sed/`, `src/diaremot/intent/`
- **Model utilities:** `src/diaremot/io/onnx_utils.py`, `src/diaremot/utils/model_paths.py`
- **Tests:** `tests/`
- **Setup scripts:** `setup.sh`, `setup.ps1`

---

## 8) Models & Data (Local‑First, Mandatory)

- Use **only** the models from `./assets/models.zip` (unzipped to `./models`).
- **Required paths (must exist):**
  - `./models/Diarization/ecapa-onnx/ecapa_tdnn.onnx`
  - `./models/Diarization/silero_vad/silero_vad.onnx`
  - `./models/Affect/ser8/model.int8.onnx`
  - `./models/Affect/VAD_dim/model.onnx`
  - `./models/Affect/sed_panns/model.onnx`
  - `./models/Affect/sed_panns/class_labels_indices.csv`
  - `./models/text_emotions/model.int8.onnx`
  - `./models/intent/model_int8.onnx`

- **Environment overrides (concrete):**
```bash
export DIAREMOT_MODEL_DIR="$(pwd)/models"
export SILERO_VAD_ONNX_PATH="$DIAREMOT_MODEL_DIR/Diarization/silero_vad/silero_vad.onnx"
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
pytest -q
```

> Always state *why* each command is being run and what success criteria you expect before execution.

---

## 5. Diagnostics & Artefacts Checklist

- `logs/run.jsonl`: Stage timing, warnings, fallback notices.
- `outputs/qc_report.json`: Audio health metrics, dependency summary, auto-tune notes (if enabled).
- `outputs/diarized_transcript_with_emotion.csv`: Confirm the 53-column schema and non-empty rows.
- `outputs/segments.jsonl`: Ensure affect, text emotion, and intent payloads are present.
- `outputs/summary.html` / `summary.pdf`: Generated when dependencies are available; note if missing.
- `/tmp/smoke/*` (for smoke runs): Validate expected artefacts and review logs for fallbacks.

Document any anomalies in your final report, even if you do not resolve them in the current session.

---

## 10) Artifacts & Success Criteria (Hard Gates)

A run is successful only if **all** of the following pass:

1. **Artifacts present:**
   - `diarized_transcript_with_emotion.csv`
   - `segments.jsonl`
   - `summary.html`
   - `speakers_summary.csv`
   - `timeline.csv`
   - `qc_report.json`
   - `events_timeline.csv` whenever the manifest advertises a timeline output (absence is only acceptable when SED timeline mode does not engage for the sample audio).
2. **CSV contract:** `diarized_transcript_with_emotion.csv` has **53 columns** including affect (V/A/D), speech emotion, text emotions top‑5 JSON, intent, SED overlaps, noise/SNR metrics, extended affect metadata (noise score, timeline events, ASR confidence/language/tokens, voice quality reliability).
2. **CSV contract:** `diarized_transcript_with_emotion.csv` has **53 columns** including affect (V/A/D), speech emotion, text emotion JSON payloads, intent, ASR confidences/tokens, SED timeline metadata, and noise/SNR metrics.
3. **All features exercised:** Quiet‑Boost, SED (ON), Diarization, ASR, Affect V/A/D, Speech Emotion 8‑class, Text Emotions 28‑class, Intent. **VAD_dim present.**
4. **No fallbacks used.** If any component indicates fallback/disablement, mark as **failure**.
5. **Lint/tests:** `ruff` and `pytest` (if tests exist) complete with **zero hard failures**.

---

## 11) Known Pitfalls / Gotchas (Concrete)

- **Directory name:** Silero VAD lives under `Diarization/silero_vad/` (one “e”; older docs with `silaro` are wrong).
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
## 6. Completion Checklist

- Deactivate the venv if desired (`deactivate`).
- Provide a final response that includes:
  - Summary of actions taken and rationale.
  - Commands executed with exit codes.
  - Key observations (logs, artefacts, diagnostics).
  - Suggested follow-ups, if any.
- Leave caches (`.cache/`, `D:/models`) untouched unless the user asked for cleanup.
- Ensure `./assets/models.zip` remains available for future offline work if local models were used.
