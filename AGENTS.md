# AGENTS.md — Local Codex IDE Agent Handbook

This document describes how the interactive Codex agent collaborates with the user when working on DiaRemot from a local IDE session. Everything below assumes the project is already installed and working; the primary job is to reason about changes, verify outcomes, and communicate clearly.

---

## 1. Role Overview

- **Plan first.** Outline intended edits, tests, or inspections and wait for explicit approval before touching the repo or running commands.
- **Execute surgically.** Apply only the approved changes, keeping edits as small and focused as possible.
- **Observe, don’t assume.** Use the existing artefacts (`logs/`, `outputs/`, `qc_report.json`, etc.) to understand pipeline behaviour before altering configuration.
- **Report concisely.** Summarise what changed, what you ran, and what you observed. Note any follow-up actions for the user.

---

## 2. Environment Assumptions

- Python 3.11, the virtual environment (`.venv`), ffmpeg, and dependencies are already in place.
- The primary model bundle lives at `D:/models`; environment variable `DIAREMOT_MODEL_DIR` should already point there. Only fall back to `./assets/models.zip` if instructed.
- Hugging Face / CT2 caches exist under `.cache/`; leave them intact unless you have a reason to rebuild.
- Network access is available for additional downloads, but unnecessary in the normal workflow.

*If any prerequisite is missing, flag it in your plan before making the change.*

---

## 3. Standard Workflow

1. **Gather context.** Review the repo state (`git status`, recent logs, pending outputs) to understand current conditions.
2. **Propose a plan.** List the smallest set of steps needed (e.g., “inspect file X,” “modify Y,” “run smoke test”). Wait for approval.
3. **Execute approved steps.** Activate the existing venv if needed (`source .venv/bin/activate` or `.\.venv\Scripts\activate`). Run only the commands that were agreed upon.
4. **Inspect diagnostics.** For pipeline runs, check `logs/run.jsonl`, `outputs/qc_report.json`, and generated artefacts. Look for fallbacks, missing columns, or dependency warnings.
5. **Communicate results.** Provide a brief summary, list of commands with exit codes, notable log excerpts, and any follow-up recommendations.

---

## 4. Useful Commands (Run Only With Approval)

```bash
# Pipeline execution on real audio
python -m diaremot.cli run --input input.wav --outdir outputs/ \
  --model-root "${DIAREMOT_MODEL_DIR:-./models}" --enable-sed --enable-affect

# Synthetic smoke test (generates demo audio)
python -m diaremot.cli smoke --outdir /tmp/smoke --enable-affect \
  --model-root "${DIAREMOT_MODEL_DIR:-./models}"

# Resume from checkpoints
python -m diaremot.cli resume --input input.wav --outdir outputs/

# Lint / tests
ruff check src/ tests/
pytest -q
```

> Always state *why* each command is being run and what success criteria you expect before execution.

---

## 5. Diagnostics & Artefacts Checklist

- `logs/run.jsonl`: Stage timing, warnings, fallback notices.
- `outputs/qc_report.json`: Audio health metrics, dependency summary, auto-tune notes (if enabled).
- `outputs/diarized_transcript_with_emotion.csv`: Confirm the 39-column schema and non-empty rows.
- `outputs/segments.jsonl`: Ensure affect, text emotion, and intent payloads are present.
- `outputs/summary.html` / `summary.pdf`: Generated when dependencies are available; note if missing.
- `/tmp/smoke/*` (for smoke runs): Validate expected artefacts and review logs for fallbacks.

Document any anomalies in your final report, even if you do not resolve them in the current session.

---

## 6. Completion Checklist

- Deactivate the venv if desired (`deactivate`).
- Provide a final response that includes:
  - Summary of actions taken and rationale.
  - Commands executed with exit codes.
  - Key observations (logs, artefacts, diagnostics).
  - Suggested follow-ups, if any.
- Leave caches (`.cache/`, `D:/models`) untouched unless the user asked for cleanup.
- Ensure `./assets/models.zip` remains available for future offline work if local models were used.
