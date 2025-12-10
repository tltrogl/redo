# Codex Environment Setup Guide

This document captures the exact steps used to bootstrap the DiaRemot Codex
worker environment for CPU-only smoke testing. Follow this sequence to reproduce
the runtime that validated the latest smoketest run.

## 1. System prerequisites

1. Start from Ubuntu 22.04 LTS (the Codex worker baseline).
2. Update the package index and install the multimedia libraries required by the
   pipeline:
   ```bash
   sudo apt-get update
   sudo apt-get install -y ffmpeg libsm6 libxext6
   ```
   * `ffmpeg` enables audio resampling, loudness normalisation, and segment
     extraction.
   * `libsm6` and `libxext6` satisfy OpenCV and Pillow runtime dependencies.

## 2. Python toolchain

1. Ensure Python 3.11 is available. Install from `deadsnakes` if the base image
   does not already include it:
   ```bash
   sudo apt-get install -y software-properties-common
   sudo add-apt-repository -y ppa:deadsnakes/ppa
   sudo apt-get update
   sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
   ```
2. Create the project virtual environment at the repo root:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```
3. Upgrade the packaging toolchain inside the venv:
   ```bash
   python -m pip install --upgrade pip wheel setuptools
   ```

## 3. Install DiaRemot dependencies

1. Install the pinned runtime stack:
   ```bash
   pip install -r requirements.txt
   ```
2. Install DiaRemot in editable mode for local development and CLI access:
   ```bash
   pip install -e .
   ```
3. (Optional) Install the CPU build of PyTorch explicitly if diagnostics report
   it missing:
   ```bash
   pip install --index-url https://download.pytorch.org/whl/cpu torch
   ```

## 4. Hydrate model assets

1. Download and validate the curated `models.zip` bundle used by Codex workers:
   ```bash
   curl -L https://github.com/tltrogl/diaremot2-ai/releases/download/v2.AI/models.zip -o models.zip
   sha256sum --check models.zip.sha256  # Expect 3cc2115f4ef7cd4f9e43cfcec376bf56ea2a8213cb760ab17b27edbc2cac206c
   unzip -q models.zip -d ./models
   ```
2. The archive expands into the alias-aware directory layout consumed by the
   pipeline (`ser8-onnx/`, `goemotions-onnx/`, `bart/`, `panns/`, `ecapa_onnx/`,
   root `silero_vad.onnx`). Case-folded lookups mean variations such as
   `GoEmotions-ONNX/` also resolve correctly.
3. Preserve the extracted directory structure; do not move ONNX files to the
   repository root.

## 5. Configure caches and environment variables

Add the following exports to your shell profile (or set them in CI) to mirror
Codex defaults:
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

These settings enforce local-first model resolution and consistent CPU threading
behaviour across runs.

## 6. Validate the environment

1. Prime the caches by running the diagnostics and smoke commands:
   ```bash
   python -m diaremot.cli diagnostics
   PYTHONPATH=src \
     HF_HOME=.cache \
     python -m diaremot.cli smoke \
       --outdir /tmp/smoke_test \
       --model-root ./models \
       --enable-affect
   ```
2. The first execution pulls Faster-Whisper distil-large-v3 (CTranslate2) and may fetch
   tokenizer metadata from Hugging Face. Subsequent runs are offline.
3. Confirm all 11 pipeline stages report `PASS` in the smoke summary and that
   `/tmp/smoke_test/` contains the CSV/JSON/HTML/PDF artefacts.

## 7. Housekeeping

- Keep `.cache/` and `./models` persisted between runs to avoid repeated
  downloads.
- Regenerate the environment by re-running the steps above after dependency
  updates or release upgrades.
