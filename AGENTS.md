# AGENTS.md — Local IDE Agent Playbook
ommunication

- **Plan before you act.** Outline the goal, proposed edits/tests, and expected outcomes. Share the plan with the user; adjust based on their feedback before running commands or touching files.
- **Stay surgical.** Modify only what the plan covers. Keep changes focused and explain your rationale.
- **Observe first.** Review existing evidence (artefacts, logs, caches) before rerunning heavy jobs. That saves time and highlights whether a rerun is even necessary.
- **Report
DiaRemot is already installed and working on this workstation. Your job as the interactive Codex agent is to understand what’s happening, make the smallest necessary adjustments, and explain every action you take. Use this guide as your operating manual.

---

## 1. Mindset & C clearly.** After each task, return a concise summary: what changed, commands + exit codes, key observations, and follow-up suggestions.

Why inspect existing artefa  xz? They capture the pipeline’s most recent behaviour—reading them tells you what actually happened (speaker counts, fallbacks, timing) without burning minutes on another run. Re-run only when you need fresh data after making changes or when artefacts are missing/invalid.

---

## 2. Workspace Overview

- **Repository root:** `D:\dia\diaremot2-on`
- **Primary virtualenv:** `.balls` (PowerShell activation: `.\.balls\Scripts\Activate.ps1`)
- **Models:** Shared bundle under `D:\models`. Only fall back to downloading `./assets/models.zip` if this drive is unavailable.
- **Caches:** Hugging Face / CT2 caches under `.cache\`; stage checkpoints under `checkpoints\`; recent run outputs in `outputs_*` directories.
- **Sample audio:** `data\sample1.mp3`, `data\sample2.mp3`, etc.
- **New modules to know:** 
  - Preprocess stack: `src/diaremot/pipeline/preprocess/`
  - Affect analyzers: `src/diaremot/affect/analyzers/`
  - Tests for these modules live under `tests/affect/` and `tests/test_preprocess_stack.py`
- **Reference docs:** When you need deeper pipeline context, consult `README.md` (pipeline overview), `DATAFLOW.md` (stage-by-stage details), `MODEL_MAP.md` (model search paths), and `agentscloud.md` (strict cloud playbook). Validate their guidance against the current code/artefacts and flag or fix any inaccuracies.

---

## 3. Typical Workflow

1. **Context check**
   - `git status -sb`
   - Inspect recent outputs (`outputs_sample*`, `/tmp/smoke` if present)
   - Review `logs/run.jsonl` for warnings or fallbacks
2. **Plan & approval**
   - Describe intended changes/tests and expected results
3. **Execute approved steps**
   - Activate venv: `.\.balls\Scripts\Activate.ps1`
   - Set caches:  
     ```powershell
     $env:HF_HOME = (Resolve-Path '.\.cache')
     $env:TRANSFORMERS_CACHE = $env:HF_HOME
     $env:DIAREMOT_MODEL_DIR = 'D:\models'
     ```
   - Run agreed commands (examples below)
4. **Verify & interpret**
   - Check outputs (CSV schema, speakers_summary, qc_report, summary.html/pdf)
   - Note speaker counts, affect coverage, any fallback warnings
5. **Summarise back to the user**
   - Actions taken, exit codes, observations, next steps

---

## 4. Command Reference (use only after approval)

```powershell
# Activate environment (PowerShell)
.\.balls\Scripts\Activate.ps1

# Full pipeline on real audio
python -m diaremot.cli run `
  --input data\sample1.mp3 `
  --outdir outputs_sample1 `
  --model-root $env:DIAREMOT_MODEL_DIR `
  --clear-cache    # add only when you need a fresh run

# Resume using checkpoints
python -m diaremot.cli resume --input data\sample1.mp3 --outdir outputs_sample1

# Synthetic smoke test
python -m diaremot.cli smoke --outdir /tmp/smoke --enable-affect --model-root $env:DIAREMOT_MODEL_DIR

# Lint & tests
ruff check src/ tests/
pytest -q
```

Before running any pipeline command, note what success looks like (e.g., “expect 4 speakers, no fallbacks”). After the run, verify that those conditions hold.

---

## 5. Diagnostics Checklist

- **`logs/run.jsonl`** – Stage timings, speaker estimates, fallback warnings. Search for `fallback`, `warn`, `issues`.
- **`outputs/*/qc_report.json`** – Confirm `dependency_ok`, inspect `warnings`, check audio health metrics (SNR, clipping).
- **`diarized_transcript_with_emotion.csv`** – 39 columns, non-empty. Check `speaker_id` diversity, `snr_db_sed`, affect scores.
- **`speakers_summary.csv`** – Verify speaker counts align with expectations.
- **`segments.jsonl`** – Ensure valence/arousal/dominance, text emotions, intent fields are populated.
- **`summary.html` / `summary.pdf`** – Confirm report generation; missing PDF often means wkhtmltopdf is absent.
- **Caches** – `.cache`, `checkpoints`, `outputs_*`. Back up before clearing so you can compare results.

Document any anomalies (missing artefacts, suspicious metrics, warnings) in your final message, even if you don’t fix them immediately.

---

## 6. Cache & Output Management

- Archive current results before clearing:  
  `Copy-Item outputs_sample1 outputs_sample1_cached`
- Clear caches only with user approval:  
  `Remove-Item -Recurse -Force .cache checkpoints`
- Keep `./assets/models.zip` (if present) and `D:\models` intact unless explicitly asked to refresh them.

---

## 7. Completion Checklist

- Deactivate venv if needed (`deactivate`).
- Provide a final response including:
  1. Summary of actions and rationale
  2. Commands with exit codes
  3. Key log/artefact observations
  4. Follow-up recommendations
- Leave the workspace in a ready-to-use state (no stray temp files, caches only cleared if necessary).

Following this playbook keeps the IDE session predictable, minimises unnecessary reruns, and ensures the user always knows what happened and why.
