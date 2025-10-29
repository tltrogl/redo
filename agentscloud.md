# AGENTSCLOUD.md — Codex Cloud Environment Agent (Strict)

This contract applies to autonomous Codex agents running DiaRemot in the cloud. Execute the workflow exactly as written—no mid-run approvals, no fallbacks, no skipped checks.

---

## 1) Autonomy & Unblocking

- **Execution model:** **Plan → Execute → Self-Verify → Continue.** Do **not** pause for approval once the run starts.
- **Unblocking mandate:** If any prerequisite is missing or broken, **detect → acquire/repair → verify** in the same run (packages, CLIs, models, directories, permissions, services, caches).
- **No fallbacks:** Any internal fallback (e.g., alternate runtimes/models/features, placeholder outputs) counts as **failure**.
- **Change style — “surgical but thorough”:** Edit the **smallest regions per file** across **all affected files** (code, config, scripts, tests, docs). Emit diffs after each edit wave.
- **Evidence every time:** Record exit codes and concise logs for every major command. Include them in the completion report.

---

## 2) Environment Assumptions (Hard Requirements)

- **OS:** Linux container/VM; treat the environment as **ephemeral and reproducible**.
- **Python:** **3.11** exactly. Create/activate a local venv.
- **CPU-only:** Use **ONNXRuntime** models where defined; **never** fall back to PyTorch or other runtimes.
- **System tools:** `ffmpeg` must be available on `PATH`.
- **Network:** **Allowed.** Use it for package/model downloads as needed.
- **Models:** **All required models are provided by `./assets/models.zip`** (SHA-256 `eb2594c5ee3e470baf7191f11109e082050c9e56fd9e3a59d76101924793df5f`). Extract to `./models` and reuse the unpacked assets. Do **not** download alternate model files.

---

## 3) Setup (Bootstrap) — Idempotent, No Placeholders

Run these commands at task start (or during environment setup). They must succeed end-to-end.

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

# 3.4 Models: download (idempotent) and unzip strictly from the provided bundle
mkdir -p ./assets
if [ ! -f ./assets/models.zip ]; then
  curl -sSLf https://github.com/tltrogl/diaremot2-ai/releases/download/v2.AI/models.zip -o ./assets/models.zip
fi
sha256sum ./assets/models.zip | grep -qi "eb2594c5ee3e470baf7191f11109e082050c9e56fd9e3a59d76101924793df5f"
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

# 3.6 Configure caches (concrete paths)
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

# Type check (required if typing present; otherwise allow non-zero but report)
mypy src/ || true

# Unit / functional tests (run if tests/ exists)
if [ -d tests ]; then
  pytest -q
fi

# Diagnostics (on demand)
python -m diaremot.cli diagnostics
```

Any hard failure must terminate the run and be surfaced in the final report.

---

## 6) Strict Smoke Check (All Features, Zero Fallbacks)

This smoke must exercise every feature (Quiet-Boost, SED ON, diarization, ASR, affect V/A/D, SER-8, text emotions 28-class, intent) with **no fallbacks**.

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
import csv, json, pathlib, sys
outdir = pathlib.Path("/tmp/smoke_strict")
manifest_path = outdir / "manifest.json"
manifest = json.load(open(manifest_path, encoding="utf-8"))

required_outputs = {
    "csv": "diarized_transcript_with_emotion.csv",
    "jsonl": "segments.jsonl",
    "timeline": "timeline.csv",
    "summary_html": "summary.html",
    "qc_report": "qc_report.json",
}
for key, default in required_outputs.items():
    path = pathlib.Path(manifest["outputs"].get(key, outdir / default))
    if not path.exists():
        print(f"Missing required artifact for {key}: {path}", file=sys.stderr)
        sys.exit(1)

csv_path = pathlib.Path(manifest["outputs"]["csv"])
with open(csv_path, newline="", encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
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
missing = [c for c in EXPECTED_COLUMNS if c not in header]
if missing or len(header) != len(EXPECTED_COLUMNS):
    print(f"CSV schema mismatch: missing={missing}, columns={len(header)}", file=sys.stderr)
    sys.exit(1)
rows = list(reader)
if not rows:
    print("CSV has zero rows", file=sys.stderr)
    sys.exit(1)

segments_jsonl = pathlib.Path(manifest["outputs"]["jsonl"])
with open(segments_jsonl, encoding="utf-8") as fh:
    affect_ok = text_ok = intent_ok = False
    for line in fh:
        obj = json.loads(line)
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
    print(f"Manifest issues detected: {issues}", file=sys.stderr)
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

Any missing artifact, schema mismatch, or fallback indicator is a **hard failure**.

---

## 7) Repo Layout & Entry Points

- **CLI:** `src/diaremot/cli.py`
- **Pipeline orchestrator:** `src/diaremot/pipeline/orchestrator.py`
- **Stage registry:** `src/diaremot/pipeline/stages/__init__.py`
- **Outputs & schema:** `src/diaremot/pipeline/outputs.py` (39 columns)
- **Affect / SED / Intent:** `src/diaremot/affect/`, `src/diaremot/sed/`, `src/diaremot/intent/`
- **Model utilities:** `src/diaremot/io/onnx_utils.py`, `src/diaremot/utils/model_paths.py`
- **Tests:** `tests/`
- **Setup scripts:** `setup.sh`, `setup.ps1`

---

## 8) Models & Data (Local-First, Mandatory)

- Use **only** the models extracted from `./assets/models.zip`.
- **Required paths (must exist):**
  - `./models/Diarization/ecapa-onnx/ecapa_tdnn.onnx`
  - `./models/Diarization/silero_vad/silero_vad.onnx`
  - `./models/Affect/ser8/model.int8.onnx`
  - `./models/Affect/VAD_dim/model.onnx`
  - `./models/Affect/sed_panns/model.onnx`
  - `./models/Affect/sed_panns/class_labels_indices.csv`
  - `./models/text_emotions/model.int8.onnx`
  - `./models/intent/model_int8.onnx`

- **Environment overrides:**
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

- **Internet:** Always allowed; use it to fetch packages or models as needed.
- **Secrets:** None required for normal operation. If secrets are introduced, fetch them only during setup and cache locally; never request them mid-run.

---

## 10) Artifacts & Success Criteria (Hard Gates)

A run is successful only if **all** criteria below are satisfied:

1. **Artifacts present:**
   - `diarized_transcript_with_emotion.csv`
   - `segments.jsonl`
   - `summary.html`
   - `speakers_summary.csv`
   - `timeline.csv`
   - `qc_report.json`
   - `events_timeline.csv` whenever the manifest exposes an events timeline (absence is acceptable only if the SED timeline path didn’t trigger).
2. **CSV contract:** `diarized_transcript_with_emotion.csv` has **39 columns** covering affect (V/A/D), SER-8, text-emotion top-5 JSON, intent, SED overlaps, noise/SNR metrics.
3. **All features exercised:** Quiet-Boost, background SED (ON), diarization, ASR, affect V/A/D, speech emotion 8-class, text emotions 28-class, intent. **VAD_dim is mandatory.**
4. **No fallbacks used:** Any fallback, placeholder, or disabled component is a failure.
5. **Lint/tests:** `ruff` and `pytest` (if tests exist) complete with **zero hard failures**.

---

## 11) Known Pitfalls / Gotchas

- **Silero directory:** The correct path is `Diarization/silero_vad/` (one “e”).
- **Quantized models:** SER8 and text-emotion models are `model.int8.onnx`; intent is `model_int8.onnx`.
- **Auto-chunking:** Long audio is automatically chunked—ensure sufficient disk space under `/tmp` and the workspace.
- **Caches:** Deleting `.cache/` or `./models/` forces redownloads; avoid unless required for the task.

---

## 12) Completion & Run Report

Include the following in the final response:

- **Summary:** Requested task, what changed, why.
- **Diffs:** Unified diffs for every modified file.
- **Commands:** Exact commands executed with exit codes.
- **Evidence:** Key log excerpts and precise artifact paths.
- **Follow-ups:** Any temporary services/caches created, plus cleanup guidance.

**Completion requirement:** All hard gates in §10 must be green. If any gate fails, report the failure, stop, and do not mark the run as complete.
