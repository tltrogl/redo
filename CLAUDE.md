# CLAUDE.md — AI Assistant Guide for DiaRemot Codebase

**Project:** DiaRemot (Speech Intelligence Pipeline)
**Version:** 2.2.0
**Language:** Python 3.11-3.12
**Last Updated:** 2025-11-22
**Purpose:** CPU-only speech intelligence pipeline for long-form audio analysis with web interface

This document provides comprehensive context for Claude AI assistants working with the DiaRemot codebase. It covers architecture, conventions, workflows, and best practices.

> **See Also:**
> - [README.md](README.md) - Complete user guide and installation
> - [DATAFLOW.md](DATAFLOW.md) - Detailed pipeline data flow documentation
> - [MODEL_MAP.md](MODEL_MAP.md) - Model inventory and search paths
> - [WEB_API_README.md](WEB_API_README.md) - Web API installation and usage
> - [GEMINI.md](GEMINI.md) - General AI assistant context
> - [AGENTS.md](AGENTS.md) - Autonomous agent setup guide

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture & Design Patterns](#architecture--design-patterns)
3. [Directory Structure](#directory-structure)
4. [Core Concepts](#core-concepts)
5. [Development Guidelines](#development-guidelines)
6. [Code Patterns & Conventions](#code-patterns--conventions)
7. [Testing Strategy](#testing-strategy)
8. [Common Tasks & Workflows](#common-tasks--workflows)
9. [Critical Constraints](#critical-constraints)
10. [Troubleshooting Guide](#troubleshooting-guide)
11. [Quick Reference](#quick-reference)

---

## Project Overview

### What is DiaRemot?

DiaRemot is a **production-ready, CPU-only speech intelligence system** that processes long-form audio (1-3 hours) into comprehensive diarized transcripts with deep affect, paralinguistic, and acoustic analysis.

**Core Capabilities:**
- **Speaker Diarization** - Silero VAD + ECAPA-TDNN embeddings with Agglomerative Hierarchical Clustering
- **Automatic Speech Recognition** - Faster-Whisper (CTranslate2) with intelligent batching
- **Emotion Analysis** - Multi-modal (audio + text): 8 speech emotions + 28 text emotions (GoEmotions)
- **Intent Classification** - Zero-shot intent detection via BART-MNLI
- **Sound Event Detection** - PANNs CNN14 for ambient sound classification (527 AudioSet classes)
- **Voice Quality Analysis** - Praat-Parselmouth metrics (jitter, shimmer, HNR, CPPS)
- **Paralinguistics** - Prosody, speech rate (WPM), pause patterns, disfluency detection
- **Persistent Speaker Registry** - Cross-file speaker tracking via embedding centroids

**Key Constraint:** CPU-only, no GPU dependencies. Optimized for commodity hardware.

**Deployment Options:**
- **CLI** - Command-line interface via Typer (main usage)
- **Programmatic API** - Python library import
- **Web API** - FastAPI REST/WebSocket server (optional, requires `[web]` extras)
- **Web UI** - Next.js 14 frontend with interactive configuration (optional)

### Technology Stack

**Core Runtime:**
- Python 3.11-3.12 (3.13+ not yet supported)
- ONNXRuntime ≥1.17 (primary inference engine)
- PyTorch 2.x (minimal fallback use)

**Audio Processing:**
- librosa 0.10.2.post1 (resampling, feature extraction)
- scipy ≥1.10,<1.14 (signal processing)
- soundfile ≥0.12 (I/O)
- Praat-Parselmouth 0.4.3 (voice quality, optional)

**ML/NLP:**
- CTranslate2 ≥4.2,<5.0 (ASR backend)
- faster-whisper ≥1.0.3 (ASR wrapper)
- transformers ≥4.40,<4.46 (HuggingFace models)
- scikit-learn ≥1.3,<1.6 (clustering)

**CLI/API:**
- Typer 0.9.0 (CLI framework)
- FastAPI ≥0.104 (Web API framework)
- Uvicorn ≥0.24 (ASGI server)
- rich 13.7.1 (console output)
- WebSockets ≥12.0 (real-time progress streaming)

---

## Architecture & Design Patterns

### Pipeline Architecture

DiaRemot uses a **modular, stage-based pipeline architecture** with:

1. **11 Sequential Processing Stages** - Each stage processes specific aspects of the audio
2. **Mixin-Based Orchestrator** - Composition pattern for component isolation
3. **State-Machine Design** - `PipelineState` dataclass passed between stages
4. **Lazy-Loading Module System** - With legacy compatibility aliases
5. **Checkpoint & Resume** - Resumable execution via digest-based caching

### 11-Stage Processing Pipeline

**Source of Truth:** `src/diaremot/pipeline/stages/__init__.py`

```python
PIPELINE_STAGES = [
    StageDefinition("dependency_check", dependency_check.run),           # 1
    StageDefinition("preprocess", preprocess.run_preprocess),            # 2
    StageDefinition("background_sed", preprocess.run_background_sed),    # 3
    StageDefinition("diarize", diarize.run),                             # 4
    StageDefinition("transcribe", asr.run),                              # 5
    StageDefinition("paralinguistics", paralinguistics.run),             # 6
    StageDefinition("affect_and_assemble", affect.run),                  # 7
    StageDefinition("overlap_interruptions", summaries.run_overlap),     # 8
    StageDefinition("conversation_analysis", summaries.run_conversation),# 9
    StageDefinition("speaker_rollups", summaries.run_speaker_rollups),   # 10
    StageDefinition("outputs", summaries.run_outputs),                   # 11
]
```

**CRITICAL:**
- Stage order is fixed and contractual
- Each stage has signature: `run(pipeline, state, config) -> None`
- Stages mutate `PipelineState` in place
- Never modify stage count or order without version bump

### Data Flow

```
Audio File (WAV/MP3/M4A)
    ↓
[1] dependency_check → Validate environment
    ↓
[2] preprocess → 16kHz mono, -20 LUFS target, auto-chunk if >30min
    ↓ {y, sr, duration_s, health, audio_sha16}
    ↓
[3] background_sed → PANNs CNN14 (global + timeline if noisy)
    ↓ {sed_info: top labels, dominant_label, noise_score, timeline?}
    ↓
[4] diarize → Silero VAD + ECAPA embeddings + AHC clustering
    ↓ {turns: [{start, end, speaker, speaker_name}], vad_unstable}
    ↓
[5] transcribe → Faster-Whisper with intelligent batching
    ↓ {norm_tx: [{start, end, speaker, text, asr_logprob_avg}]}
    ↓
[6] paralinguistics → Praat (jitter/shimmer/HNR/CPPS) + prosody
    ↓ {para_map: {seg_idx: {wpm, f0_mean_hz, vq_jitter_pct, ...}}}
    ↓
[7] affect_and_assemble → Audio (VAD+SER8) + Text (GoEmotions+BART-MNLI)
    ↓ {segments_final: 53 columns per segment}
    ↓
[8] overlap_interruptions → Detect overlaps + classify interruptions
    ↓ {overlap_stats, per_speaker_interrupts}
    ↓
[9] conversation_analysis → Turn-taking + dominance + flow metrics
    ↓ {conv_metrics: ConversationMetrics}
    ↓
[10] speaker_rollups → Aggregate per-speaker stats
    ↓ {speakers_summary: [{speaker, duration, affect, voice_quality, ...}]}
    ↓
[11] outputs → Write CSV/JSONL/HTML/PDF/QC reports
```

**Detailed Documentation:** See [DATAFLOW.md](DATAFLOW.md) for complete stage-by-stage specifications.

### Design Patterns Used

#### 1. Mixin Composition Pattern

**File:** `src/diaremot/pipeline/orchestrator.py`

```python
class AudioAnalysisPipelineV2(
    ComponentFactoryMixin,    # Component initialization
    AffectMixin,              # Affect analysis helpers
    ParalinguisticsMixin,     # Voice quality extraction
    OutputMixin,              # Report generation
):
    """Main orchestrator using multiple mixins for separation of concerns."""
```

**When to use:** Adding new component categories or capabilities to the pipeline.

#### 2. State Machine Pattern

**File:** `src/diaremot/pipeline/stages/base.py`

```python
@dataclass
class PipelineState:
    """Mutable state passed between pipeline stages."""
    input_audio_path: str
    out_dir: Path
    y: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    sr: int = 16000
    # ... 40+ fields total
```

**When to use:** Adding new state fields that stages need to share.

#### 3. Lazy Loading Pattern

**File:** `src/diaremot/__init__.py`

```python
_LEGACY_MODULE_ALIASES = {
    "audio_pipeline_core": "diaremot.pipeline.audio_pipeline_core",
    "emotion_analysis": "diaremot.affect.emotion_analysis",
    # ... 20+ aliases
}

def __getattr__(name):
    """Lazy-load modules and provide legacy aliases."""
    if name in _LEGACY_MODULE_ALIASES:
        return importlib.import_module(_LEGACY_MODULE_ALIASES[name])
    raise AttributeError(f"module 'diaremot' has no attribute '{name}'")
```

**When to use:** Adding new modules while maintaining backward compatibility.

#### 4. Digest-Based Caching Pattern

**File:** `src/diaremot/pipeline/audio_pipeline_core.py`

```python
def matches_pipeline_cache(cache_meta: dict, audio_sha16: str, pp_sig: dict) -> bool:
    """Validate cache via version + audio hash + preprocessing signature."""
    return (
        cache_meta.get("version") == CACHE_VERSION
        and cache_meta.get("audio_sha16") == audio_sha16
        and cache_meta.get("pp_sig") == pp_sig
    )
```

**When to use:** Adding new cacheable stages or modifying cache validation.

#### 5. Structured Error Handling

**File:** `src/diaremot/pipeline/errors.py`

```python
class StageExecutionError(Exception):
    """Raised when a pipeline stage fails."""
    def __init__(self, stage: str, message: str, original_exception: Exception | None = None):
        self.stage = stage
        self.message = message
        self.original_exception = original_exception
        super().__init__(f"[{stage}] {message}")
```

**When to use:** Raising errors in stage execution or component initialization.

#### 6. Factory Pattern

**File:** `src/diaremot/pipeline/core/component_factory.py`

```python
class ComponentFactoryMixin:
    """Provides component initialization with structured error handling."""

    def _init_transcriber(self, config: dict) -> Transcriber:
        """Initialize ASR transcriber with backend detection."""
        try:
            return Transcriber(config)
        except Exception as e:
            raise StageExecutionError(
                stage="transcribe",
                message=f"Failed to initialize transcriber: {e}",
                original_exception=e
            )
```

**When to use:** Adding new components that require initialization.

---

## Directory Structure

### Core Package Structure

```
redo/
├── src/
│   ├── audio_pipeline_core.py          # Legacy location (transitional)
│   └── diaremot/                       # Main package (17 packages total)
│       ├── __init__.py                 # Lazy loading factory functions
│       ├── cli.py                      # Typer-based CLI (main entry point)
│       ├── api.py                      # FastAPI wrapper for Cloud Run
│       │
│       ├── affect/                     # Emotion & affect analysis
│       │   ├── analyzers/             # Modular analyzers (speech, text, VAD, intent)
│       │   ├── paralinguistics/       # Voice quality & prosody extraction package
│       │   ├── emotion_analyzer.py    # Orchestrator for affect models
│       │   ├── ser_onnx.py           # Speech emotion recognition (8-class)
│       │   └── sed_panns.py          # Sound event detection (PANNs CNN14)
│       │
│       ├── io/                        # Input/output utilities
│       │   ├── download_utils.py     # Model download helpers
│       │   ├── onnx_utils.py         # ONNX model loading utilities
│       │   └── speaker_registry_manager.py  # Cross-file speaker tracking
│       │
│       ├── pipeline/                  # Core pipeline orchestration
│       │   ├── core/                 # Mixin components
│       │   │   ├── affect_mixin.py
│       │   │   ├── component_factory.py
│       │   │   ├── output_mixin.py
│       │   │   └── paralinguistics_mixin.py
│       │   ├── diarization/          # Speaker diarization package
│       │   │   ├── pipeline.py       # Main diarization orchestrator
│       │   │   ├── vad.py           # Silero VAD
│       │   │   ├── embeddings.py    # ECAPA-TDNN speaker embeddings
│       │   │   ├── clustering.py    # Agglomerative Hierarchical Clustering
│       │   │   └── registry.py      # Persistent speaker tracking
│       │   ├── preprocess/           # Audio preprocessing package
│       │   │   ├── chain.py         # Signal chain with spectral caching
│       │   │   ├── chunking.py      # Chunk lifecycle management
│       │   │   ├── config.py        # Preprocessing configuration
│       │   │   ├── denoise.py       # Denoising primitives
│       │   │   └── io.py            # Disk I/O
│       │   ├── runtime/              # Runtime environment & session management
│       │   ├── stages/               # 11 pipeline stages
│       │   │   ├── __init__.py      # PIPELINE_STAGES registry
│       │   │   ├── base.py          # PipelineState definition
│       │   │   ├── dependency_check.py
│       │   │   ├── preprocess.py
│       │   │   ├── diarize.py
│       │   │   ├── asr.py
│       │   │   ├── paralinguistics.py
│       │   │   ├── affect.py
│       │   │   └── summaries.py
│       │   ├── transcription/        # ASR package
│       │   │   ├── transcription_module.py  # Transcriber façade
│       │   │   ├── backends.py      # Backend detection & environment guards
│       │   │   ├── models.py        # Data classes & utilities
│       │   │   ├── postprocess.py   # Transcript redistribution helpers
│       │   │   └── scheduler.py     # Async engine & batching heuristics
│       │   ├── orchestrator.py       # Main pipeline class (AudioAnalysisPipelineV2)
│       │   ├── config.py            # Configuration & validation
│       │   ├── errors.py            # Structured error classes
│       │   ├── outputs.py           # 53-column CSV schema definition
│       │   └── audio_pipeline_core.py  # Main pipeline API
│       │
│       ├── sed/                      # Sound event detection variants
│       ├── summaries/                # Report generation
│       │   ├── conversation_analysis.py
│       │   ├── html_summary_generator.py
│       │   ├── pdf_summary_generator.py
│       │   └── speakers_summary_builder.py
│       ├── tools/                    # Dependency checks, diagnostics
│       │   └── deps_check.py
│       ├── utils/                    # Utilities
│       │   ├── model_paths.py       # Model path resolution
│       │   └── hash.py              # Hashing utilities
│       │
│       └── web/                      # Web API and server (optional)
│           ├── api/                 # FastAPI application
│           │   ├── app.py          # Main FastAPI app and endpoints
│           │   ├── models.py       # Pydantic models for API
│           │   ├── routes/         # API route modules
│           │   └── websocket.py    # WebSocket handlers for progress
│           ├── config_schema.py    # Configuration schema for web UI
│           └── server.py           # Development server launcher
│
├── tests/                            # Test suite (mirrors src/ structure)
│   ├── conftest.py                  # Test configuration (adds src/ to path)
│   ├── affect/                      # Affect analyzer tests
│   ├── pipeline/                    # Pipeline stage tests
│   ├── transcription/               # ASR backend tests
│   ├── test_diarization.py
│   ├── test_paralinguistics.py
│   └── test_outputs_transcript.py
│
├── scripts/                          # Utility scripts
│   ├── diaremot_run.sh             # Production wrapper (auto-sets threads)
│   ├── affect_only.py              # Standalone affect analysis
│   ├── audit_models_all.py         # Model inventory checker
│   └── preprocess_audio.py         # Standalone preprocessing CLI
│
├── docs/                             # Extended documentation
│   ├── DATAFLOW.md                 # Stage-by-stage data flow
│   ├── pipeline_stage_analysis.md  # Stage responsibilities & rationale
│   ├── pipeline_optimization_opportunities.md
│   └── runs/                       # Example execution logs
│
├── frontend/                         # Web frontend (Next.js 14)
│   └── frontend/                    # Next.js application root
│       ├── app/                    # Next.js app directory
│       ├── components/             # React components
│       ├── public/                 # Static assets
│       ├── package.json            # Frontend dependencies
│       └── next.config.mjs         # Next.js configuration
│
├── data/                             # Sample audio files
├── checkpoints/                      # Pipeline checkpoints
├── outputs/                          # Default output directory
├── logs/                             # Log files
├── .cache/                          # HuggingFace/model cache
│
├── pyproject.toml                   # Package configuration
├── requirements.txt                 # Python dependencies (200+ packages)
├── requirements.in                  # Human-edited pinned versions
├── constraints.txt                  # Version constraints
│
├── Dockerfile                       # Primary container definition
├── Dockerfile.cloudrun             # Cloud Run specific
├── cloudbuild.yaml                 # Cloud Build configuration
├── deploy-cloudrun.sh/.ps1         # Deployment scripts
├── setup.sh/.ps1                   # Local environment setup
│
├── README.md                        # Complete user guide (1200 lines)
├── CLAUDE.md                        # This file
├── GEMINI.md                        # General AI assistant context
├── AGENTS.md                        # Agent setup guide
├── DATAFLOW.md                      # Pipeline data flow
├── MODEL_MAP.md                     # Model inventory
├── WEB_API_README.md                # Web API installation and usage
├── WEB_APP_PROGRESS.md              # Frontend development progress
└── CLOUD_BUILD_GUIDE.md            # Cloud build instructions
```

### Key Directories Explained

#### `/src/diaremot/pipeline/`
**Purpose:** Core pipeline orchestration and stage execution
- **orchestrator.py** - Main `AudioAnalysisPipelineV2` class with mixin composition
- **stages/** - 11 stage modules, each implementing `run(pipeline, state, config)`
- **config.py** - Configuration validation, defaults, dependency checks
- **outputs.py** - 53-column CSV schema definition (CONTRACTUAL)
- **errors.py** - Structured exception classes

#### `/src/diaremot/affect/`
**Purpose:** Multimodal emotion, intent, and affect analysis
- **analyzers/** - Specialized components (speech, text, VAD, intent)
- **paralinguistics/** - Full package for prosody, voice quality (Praat integration)
- **emotion_analyzer.py** - Orchestrates all affect models
- All models support ONNX inference for production performance

#### `/src/diaremot/pipeline/diarization/`
**Purpose:** Speaker diarization subsystem
- **pipeline.py** - Main diarization orchestrator
- **vad.py** - Voice activity detection (Silero VAD)
- **embeddings.py** - ECAPA-TDNN speaker embeddings
- **clustering.py** - Agglomerative Hierarchical Clustering (AHC)
- **registry.py** - Persistent speaker tracking across files

#### `/src/diaremot/pipeline/transcription/`
**Purpose:** Modular ASR subsystem
- **transcription_module.py** - `Transcriber` façade
- **backends.py** - Backend detection & environment guards
- **scheduler.py** - Async batching engine for Faster-Whisper
- **postprocess.py** - Transcript redistribution helpers

#### `/tests/`
**Purpose:** Comprehensive test suite
- Mirrors `src/` structure
- `conftest.py` adds `src/` to Python path
- Stage-specific tests in `pipeline/`
- Component tests in `affect/`, `transcription/`, etc.

#### `/src/diaremot/web/`
**Purpose:** Web API and server (optional feature)
- **api/app.py** - Main FastAPI application with REST endpoints
- **api/models.py** - Pydantic models for request/response validation
- **api/routes/** - Modular API route handlers
- **api/websocket.py** - WebSocket endpoints for real-time progress streaming
- **config_schema.py** - Configuration schema for 70+ pipeline parameters
- **server.py** - Development server launcher script

**Installation:** Requires `pip install -e ".[web]"` for FastAPI dependencies

#### `/frontend/`
**Purpose:** Next.js 14 web interface (optional)
- **TypeScript + Tailwind CSS** modern React application
- **Interactive configuration panel** with 70+ adjustable parameters
- **Real-time processing updates** via WebSocket connection
- **File upload and download** for audio processing
- **Responsive design** for desktop and mobile

**Setup:** `cd frontend/frontend && npm install`

---

## Core Concepts

### 1. PipelineState

**File:** `src/diaremot/pipeline/stages/base.py`

The `PipelineState` dataclass is the **central data structure** passed between all stages. It's mutable and accumulates results as it flows through the pipeline.

**Key Fields:**
```python
@dataclass
class PipelineState:
    # Input
    input_audio_path: str
    out_dir: Path

    # Preprocessing output
    y: np.ndarray                    # Preprocessed audio waveform
    sr: int = 16000                  # Sample rate
    duration_s: float = 0.0
    audio_sha16: str = ""            # Audio hash for caching
    health: dict = field(default_factory=dict)

    # Diarization output
    turns: list[dict] = field(default_factory=list)
    vad_unstable: bool = False

    # Transcription output
    norm_tx: list[dict] = field(default_factory=list)

    # Paralinguistics output
    para_map: dict = field(default_factory=dict)

    # Affect output
    segments_final: list[dict] = field(default_factory=list)

    # SED output
    sed_info: dict = field(default_factory=dict)

    # Conversation analysis output
    overlap_stats: dict = field(default_factory=dict)
    per_speaker_interrupts: dict = field(default_factory=dict)
    conv_metrics: dict = field(default_factory=dict)
    speakers_summary: list[dict] = field(default_factory=list)

    # ... 40+ total fields
```

**Usage Pattern:**
```python
def run(pipeline, state: PipelineState, config: dict) -> None:
    """Stage execution pattern."""
    # 1. Read from state
    audio = state.y
    sr = state.sr

    # 2. Perform processing
    results = process_audio(audio, sr)

    # 3. Update state
    state.my_stage_output = results
```

### 2. CSV Schema Contract

**File:** `src/diaremot/pipeline/outputs.py`

The output CSV schema is **CONTRACTUAL** - column order and count are fixed. Currently **53 columns**.

**Schema Version:** Tracked via `CACHE_VERSION` constant.

**Critical Columns (in order):**
1. `file_id` - File identifier
2. `start` - Segment start time (seconds)
3. `end` - Segment end time (seconds)
4. `speaker_id` - Internal speaker ID
5. `speaker_name` - Human-readable speaker name
6. `text` - Transcribed text
7. `valence` - Valence (-1 to +1)
8. `arousal` - Arousal (-1 to +1)
9. `dominance` - Dominance (-1 to +1)
10. `emotion_top` - Top speech emotion label
... (see README.md or outputs.py for complete list)

**IMPORTANT:**
- Never modify column order
- Never insert columns (only append)
- Always verify column count after changes
- Schema changes require version bump and migration plan

### 3. Model Search & Loading

**File:** `src/diaremot/utils/model_paths.py`

DiaRemot uses a **priority-based model discovery system**.

**Search Priority:**
1. Explicit environment variable (e.g., `ECAPA_ONNX_PATH`)
2. `$DIAREMOT_MODEL_DIR`
3. `./models` (current directory)
4. `~/models` (user home)
5. OS-specific defaults:
   - **Windows:** `D:/models`, `D:/diaremot/diaremot2-1/models`
   - **Linux/Mac:** `/models`, `/opt/diaremot/models`, `/srv/models`

**Example Search Pattern (ECAPA):**
```python
# For each model root, try these relative paths:
search_paths = [
    "ecapa_onnx/ecapa_tdnn.onnx",
    "Diarization/ecapa-onnx/ecapa_tdnn.onnx",  # <-- DOCUMENTED PATH
    "ecapa_tdnn.onnx",
]

# Model roots (in priority order)
roots = [
    os.getenv("DIAREMOT_MODEL_DIR"),
    "./models",
    str(Path.home() / "models"),
    "D:/models",  # Windows
    "/srv/models",  # Linux
]

# First existing file wins
```

**When adding new models:**
- Add search paths to `model_paths.py`
- Document environment variable override
- Support both ONNX and PyTorch fallback
- Test discovery from multiple roots

### 4. Caching & Checkpointing

**Files:**
- `src/diaremot/pipeline/audio_pipeline_core.py`
- `src/diaremot/pipeline/pipeline_checkpoint_system.py`

**Three-Level Cache Validation:**
1. **Cache version** - `CACHE_VERSION = "v3"`
2. **Audio hash** - SHA-16 digest of preprocessed audio
3. **Preprocessing signature** - Config-based digest

**Cache Files (per audio file):**
```
.cache/{audio_sha16}/
├── preprocessed_audio.npy     # Audio buffer (float32)
├── preprocessed.meta.json     # Version + audio_sha16 + pp_signature + health
├── diar.json                  # Diarization turns + embeddings
├── tx.json                    # Transcription segments
└── preprocessed.npz (legacy)  # Back-compat loader remains supported
```

**Cache Hit Conditions:**
```python
cache_valid = (
    cache_meta["version"] == CACHE_VERSION
    and cache_meta["audio_sha16"] == audio_sha16
    and cache_meta["pp_sig"] == pp_sig
)
```

**Checkpoint Files (for resume):**
```
checkpoints/{session_id}/
├── state_after_stage_N.pkl  # PipelineState snapshot
└── metadata.json           # Stage progress tracking
```

### 5. Affect Memory Windows (v2.2.1 Optimization)

**Pattern:** Use `memoryview` for zero-copy audio slicing

```python
def _affect_unified(audio: memoryview | np.ndarray, ...):
    """Accept buffer-compatible iterables to avoid intermediate copies."""
    # Long-form jobs reuse the same buffer across VAD, SER, and intent analyzers
```

**Benefits:**
- Reduces peak RSS for multi-hour recordings
- Avoids materializing new NumPy arrays for every segment
- Supports generator-based streaming for advanced callers

---

## Development Guidelines

### Code Style & Quality Tools

#### 1. Ruff (Linting & Formatting)

**Configuration:** `pyproject.toml`

```toml
[tool.ruff]
line-length = 100
target-version = "py311"
extend-select = ["E", "F", "I", "UP"]
ignore = ["E501"]  # Line length enforcement disabled
```

**Commands:**
```bash
# Lint code
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/

# Format code
ruff format src/ tests/
```

#### 2. mypy (Type Checking)

**Configuration:** `pyproject.toml`

```toml
[tool.mypy]
python_version = "3.11"
strict = false
warn_return_any = true
warn_unused_configs = true
exclude = ["dist/", "build/"]
```

**Commands:**
```bash
# Type check
mypy src/

# Specific module
mypy src/diaremot/pipeline/orchestrator.py
```

#### 3. pytest (Testing)

**Configuration:** `pyproject.toml`

```toml
[tool.pytest.ini_options]
addopts = "-q"
testpaths = ["tests"]
```

**Commands:**
```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_diarization.py::test_speaker_count -v

# With coverage
pytest tests/ --cov=diaremot --cov-report=html

# Specific marker
pytest tests/ -m "not slow"
```

### Import Conventions

**Strict ordering:**
```python
from __future__ import annotations  # Always first for PEP 563

# 1. Standard library
import logging
import time
from pathlib import Path
from typing import Any

# 2. Third-party
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

# 3. Relative imports (same package)
from . import speaker_diarization
from .config import (  # Multi-line grouped imports
    DEFAULT_PIPELINE_CONFIG,
    PipelineConfig,
)

# 4. Absolute imports (parent package)
from diaremot.io.onnx_utils import load_onnx_model
from diaremot.utils.hash import compute_audio_hash
```

### Type Annotation Style

**Use Python 3.11+ features:**
```python
# PEP 604 union syntax (preferred)
def process(data: dict[str, Any] | None) -> list[str]:
    ...

# Generic types without imports (Python 3.9+)
results: dict[str, list[float]] = {}

# Forward references for circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diaremot.pipeline.orchestrator import AudioAnalysisPipelineV2
```

### Docstring Conventions

**Use Google-style docstrings:**
```python
def run_stage(
    pipeline: AudioAnalysisPipelineV2,
    state: PipelineState,
    config: dict[str, Any]
) -> None:
    """Execute the diarization stage.

    Performs speaker segmentation using Silero VAD for voice activity
    detection, ECAPA-TDNN for speaker embeddings, and Agglomerative
    Hierarchical Clustering for speaker assignment.

    Args:
        pipeline: Pipeline orchestrator with initialized components.
        state: Mutable pipeline state with audio data.
        config: Configuration dictionary with VAD and clustering parameters.

    Raises:
        StageExecutionError: If VAD or embedding extraction fails.

    Note:
        Mutates `state.turns` and `state.vad_unstable` in place.
    """
```

### Error Handling Patterns

**Use structured exceptions:**
```python
from diaremot.pipeline.errors import StageExecutionError

# In stage execution
try:
    result = expensive_operation()
except Exception as e:
    raise StageExecutionError(
        stage="my_stage",
        message=f"Operation failed: {e}",
        original_exception=e
    )

# In component initialization
try:
    model = load_model(path)
except FileNotFoundError as e:
    raise StageExecutionError(
        stage="component_init",
        message=f"Model not found at {path}",
        original_exception=e
    )
```

**Graceful degradation pattern:**
```python
# Warn and continue rather than fail hard
try:
    optional_metric = compute_advanced_metric(data)
    state.metrics["advanced"] = optional_metric
except Exception as e:
    logger.warning("Advanced metric computation failed: %s", e)
    state.metrics["advanced"] = None  # Neutral fallback
```

### Logging Conventions

**Consistent logging with context:**
```python
import logging

logger = logging.getLogger(__name__)

# Use f-strings sparingly, prefer % formatting for performance
logger.info("Processing %s with %d threads", file_path, thread_count)
logger.debug("Config: %s", config)
logger.warning("Low confidence detected: %.2f", confidence)
logger.error("Failed to load model from %s: %s", model_path, exc)

# Include context in exception logs
try:
    result = operation()
except Exception as e:
    logger.exception("Operation failed for file %s", file_path)
    raise
```

---

## Code Patterns & Conventions

### 1. Stage Implementation Pattern

**Template for new stages:**

```python
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from diaremot.pipeline.errors import StageExecutionError

if TYPE_CHECKING:
    from diaremot.pipeline.orchestrator import AudioAnalysisPipelineV2
    from diaremot.pipeline.stages.base import PipelineState

logger = logging.getLogger(__name__)


def run(
    pipeline: AudioAnalysisPipelineV2,
    state: PipelineState,
    config: dict[str, Any]
) -> None:
    """Execute the my_stage stage.

    Args:
        pipeline: Pipeline orchestrator with initialized components.
        state: Mutable pipeline state.
        config: Configuration dictionary.

    Raises:
        StageExecutionError: If stage execution fails.
    """
    logger.info("Starting my_stage")

    # 1. Validate prerequisites
    if not state.y.size:
        raise StageExecutionError(
            stage="my_stage",
            message="No audio data available"
        )

    # 2. Extract config
    param1 = config.get("param1", default_value)
    param2 = config.get("param2", default_value)

    # 3. Perform processing
    try:
        results = process_data(state.y, param1, param2)
    except Exception as e:
        raise StageExecutionError(
            stage="my_stage",
            message=f"Processing failed: {e}",
            original_exception=e
        )

    # 4. Update state
    state.my_stage_output = results

    logger.info("Completed my_stage: %d results", len(results))
```

### 2. Dataclass Pattern for State Objects

**Use dataclasses with defaults:**
```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class MyStateObject:
    """State object for stage X."""

    # Required fields (no default)
    input_path: str
    output_dir: str

    # Optional fields with defaults
    sample_rate: int = 16000
    threshold: float = 0.35

    # Mutable defaults (use field with factory)
    segments: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    audio: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
```

### 3. Configuration Validation Pattern

**Use Pydantic for config validation:**
```python
from pydantic import BaseModel, Field, field_validator

class StageConfig(BaseModel):
    """Configuration for my stage."""

    threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    min_duration: float = Field(default=0.3, gt=0.0)
    max_speakers: int = Field(default=10, gt=0)
    enable_feature: bool = Field(default=True)

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Ensure threshold is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Threshold must be in [0.0, 1.0], got {v}")
        return v
```

### 4. Mixin Pattern for Orchestrator

**Add capabilities via mixins:**
```python
class MyCapabilityMixin:
    """Provides X capability to pipeline orchestrator."""

    def my_helper_method(self, data: np.ndarray) -> dict:
        """Perform X operation."""
        # Implementation
        return results

    def _private_helper(self, x: float) -> float:
        """Internal helper (prefix with _)."""
        return x * 2.0


class AudioAnalysisPipelineV2(
    ComponentFactoryMixin,
    AffectMixin,
    ParalinguisticsMixin,
    OutputMixin,
    MyCapabilityMixin,  # <-- Add new mixin
):
    """Main orchestrator."""
```

### 5. Model Loading Pattern

**Support ONNX + PyTorch fallback:**
```python
import logging
from pathlib import Path

import onnxruntime as ort
import torch

logger = logging.getLogger(__name__)


def load_my_model(model_path: str | Path) -> ort.InferenceSession | torch.nn.Module:
    """Load model with ONNX preference and PyTorch fallback.

    Args:
        model_path: Path to model directory or ONNX file.

    Returns:
        Loaded model (ONNX session or PyTorch module).

    Raises:
        FileNotFoundError: If model not found at any search path.
    """
    model_path = Path(model_path)

    # Try ONNX first
    onnx_candidates = [
        model_path / "model.onnx",
        model_path / "model.int8.onnx",
        model_path if model_path.suffix == ".onnx" else None,
    ]

    for onnx_path in onnx_candidates:
        if onnx_path and onnx_path.exists():
            logger.info("Loading ONNX model from %s", onnx_path)
            session = ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"]
            )
            return session

    # Fallback to PyTorch
    logger.warning("ONNX model not found, falling back to PyTorch (slower)")
    try:
        model = torch.jit.load(str(model_path / "model.pt"))
        model.eval()
        return model
    except Exception as e:
        raise FileNotFoundError(f"No valid model found at {model_path}") from e
```

### 6. Zero-Copy Audio Slicing Pattern

**Use memoryview for efficiency:**
```python
import numpy as np

def process_segments(
    audio: np.ndarray,
    segments: list[dict],
    sr: int
) -> list[dict]:
    """Process audio segments with zero-copy slicing.

    Args:
        audio: Full audio waveform.
        segments: List of {start, end} dicts.
        sr: Sample rate.

    Returns:
        Processed segments with results.
    """
    # Create memoryview for zero-copy slicing
    audio_view = memoryview(audio)

    results = []
    for seg in segments:
        start_idx = int(seg["start"] * sr)
        end_idx = int(seg["end"] * sr)

        # Slice without copying
        seg_audio = audio_view[start_idx:end_idx]

        # Process (convert to numpy only when needed)
        result = analyze(np.asarray(seg_audio))
        results.append(result)

    return results
```

---

## Testing Strategy

### Test Organization

**Mirror source structure:**
```
tests/
├── conftest.py                     # Shared fixtures
├── affect/
│   ├── test_emotion_analyzer.py   # Affect tests
│   └── test_ser_onnx.py
├── pipeline/
│   ├── test_affect_stage.py       # Stage integration tests
│   ├── test_diarize_stage.py
│   └── test_orchestrator.py
└── test_diarization.py             # Top-level tests
```

### Fixture Patterns

**conftest.py setup:**
```python
import sys
from pathlib import Path

import pytest
import numpy as np

# Add src/ to path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture
def sample_audio() -> tuple[np.ndarray, int]:
    """Generate sample audio for testing."""
    sr = 16000
    duration = 5.0  # seconds
    samples = int(sr * duration)
    audio = np.random.randn(samples).astype(np.float32) * 0.1
    return audio, sr


@pytest.fixture
def mock_pipeline_state(tmp_path):
    """Create mock PipelineState for testing."""
    from diaremot.pipeline.stages.base import PipelineState

    return PipelineState(
        input_audio_path="test.wav",
        out_dir=tmp_path,
        sr=16000,
    )
```

### Mock Patterns

**Mocking external dependencies:**
```python
import pytest
from unittest.mock import Mock, patch, MagicMock


class MockONNXModel:
    """Mock ONNX model for testing."""

    def __init__(self, output_shape):
        self.output_shape = output_shape
        self.call_count = 0

    def run(self, output_names, input_feed):
        """Mock inference."""
        self.call_count += 1
        # Return dummy output matching expected shape
        return [np.zeros(self.output_shape)]


def test_model_inference_with_mock():
    """Test inference using mock model."""
    mock_model = MockONNXModel(output_shape=(1, 8))

    # Use mock
    result = mock_model.run(None, {"input": np.zeros((1, 100))})

    assert mock_model.call_count == 1
    assert result[0].shape == (1, 8)
```

### Parametrized Tests

**Test multiple scenarios:**
```python
import pytest


@pytest.mark.parametrize("threshold,expected_count", [
    (0.25, 3),
    (0.35, 2),
    (0.45, 1),
])
def test_vad_with_thresholds(threshold, expected_count):
    """Test VAD with different thresholds."""
    from diaremot.pipeline.diarization.vad import detect_speech

    audio = generate_test_audio()
    segments = detect_speech(audio, threshold=threshold)

    assert len(segments) == expected_count
```

### Integration Test Pattern

**Test full stage execution:**
```python
def test_diarize_stage_integration(tmp_path, sample_audio):
    """Integration test for diarization stage."""
    from diaremot.pipeline.stages import diarize
    from diaremot.pipeline.stages.base import PipelineState
    from diaremot.pipeline.orchestrator import AudioAnalysisPipelineV2

    # Setup
    audio, sr = sample_audio
    state = PipelineState(
        input_audio_path="test.wav",
        out_dir=tmp_path,
        y=audio,
        sr=sr,
    )

    config = {
        "vad_threshold": 0.35,
        "ahc_distance_threshold": 0.15,
    }

    # Create minimal pipeline
    pipeline = AudioAnalysisPipelineV2()

    # Execute stage
    diarize.run(pipeline, state, config)

    # Verify
    assert len(state.turns) > 0
    assert all("speaker" in turn for turn in state.turns)
    assert all("start" in turn and "end" in turn for turn in state.turns)
```

---

## Common Tasks & Workflows

### 1. Adding a New Pipeline Stage

**Checklist:**

1. **Create stage module:** `src/diaremot/pipeline/stages/my_stage.py`
   ```python
   def run(pipeline, state, config) -> None:
       """Execute my_stage."""
       # Implementation
   ```

2. **Import in stages/__init__.py:**
   ```python
   from . import my_stage

   PIPELINE_STAGES = [
       # ... existing stages
       StageDefinition("my_stage", my_stage.run),
       # ... remaining stages
   ]
   ```

3. **Add state fields if needed:** `stages/base.py`
   ```python
   @dataclass
   class PipelineState:
       # ... existing fields
       my_stage_output: dict = field(default_factory=dict)
   ```

4. **Write integration test:** `tests/pipeline/test_my_stage.py`
   ```python
   def test_my_stage_execution():
       # Test implementation
   ```

5. **Update documentation:**
   - Update stage count in README.md
   - Add stage description
   - Update this CLAUDE.md
   - Update DATAFLOW.md with stage details

6. **Verify:**
   ```bash
   pytest tests/pipeline/test_my_stage.py -v
   ruff check src/diaremot/pipeline/stages/my_stage.py
   ```

### 2. Adding a New CSV Column

**CRITICAL:** Only append, never insert!

**Checklist:**

1. **Append to SEGMENT_COLUMNS:** `src/diaremot/pipeline/outputs.py`
   ```python
   SEGMENT_COLUMNS = [
       # ... existing 53 columns
       "my_new_column",  # 54
   ]
   ```

2. **Add default value:** `outputs.py::ensure_segment_keys()`
   ```python
   def ensure_segment_keys(seg: dict) -> dict:
       """Ensure all required keys present."""
       defaults = {
           # ... existing defaults
           "my_new_column": None,  # or appropriate default
       }
       return {**defaults, **seg}
   ```

3. **Populate in appropriate stage:**
   ```python
   # In affect_and_assemble or relevant stage
   for seg in state.segments_final:
       seg["my_new_column"] = compute_value(seg)
   ```

4. **Write unit test:** `tests/test_outputs_transcript.py`
   ```python
   def test_segment_has_new_column():
       seg = create_test_segment()
       assert "my_new_column" in seg
   ```

5. **Update documentation:**
   - Update column count (53 → 54) everywhere
   - Add to README.md CSV schema section
   - Document migration for existing CSVs
   - Update this file

6. **Verify schema:**
   ```bash
   python -c "from diaremot.pipeline.outputs import SEGMENT_COLUMNS; print(len(SEGMENT_COLUMNS))"
   # Should print: 54
   ```

### 3. Adding a New Model

**Checklist:**

1. **Add ONNX loading helper:** `src/diaremot/io/onnx_utils.py`
   ```python
   def load_my_model(model_dir: Path) -> ort.InferenceSession:
       """Load my model ONNX."""
       model_path = model_dir / "model.onnx"
       return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
   ```

2. **Add model search paths:** `src/diaremot/utils/model_paths.py`
   ```python
   MY_MODEL_SEARCH_PATHS = [
       "my_model/model.onnx",
       "MyModel/model.onnx",
       "model.onnx",
   ]
   ```

3. **Add PyTorch fallback (optional):**
   ```python
   def load_my_model_pytorch() -> torch.nn.Module:
       """Fallback PyTorch loader."""
       from transformers import AutoModel
       return AutoModel.from_pretrained("org/my-model")
   ```

4. **Document model:**
   - Add to README.md model section
   - Add to MODEL_MAP.md
   - Document environment variable override
   - Document model source and license

5. **Test both paths:**
   ```python
   def test_load_my_model_onnx():
       model = load_my_model(onnx_path)
       assert isinstance(model, ort.InferenceSession)

   def test_load_my_model_pytorch():
       model = load_my_model_pytorch()
       assert isinstance(model, torch.nn.Module)
   ```

### 4. Running the Pipeline

**Basic usage:**
```bash
# Standard processing
python -m diaremot.cli run --input audio/sample.mp3 --outdir outputs/

# Fast mode (int8 quantization)
python -m diaremot.cli run -i audio/sample.mp3 -o outputs/ --asr-compute-type int8

# With custom VAD threshold
python -m diaremot.cli run -i audio/sample.mp3 -o outputs/ --vad-threshold 0.25

# Disable optional stages
python -m diaremot.cli run -i audio/sample.mp3 -o outputs/ --disable-sed --disable-affect

# Use profile
python -m diaremot.cli run -i audio/sample.mp3 -o outputs/ --profile fast

# Resume from checkpoint
python -m diaremot.cli resume -i audio/sample.mp3 -o outputs/

# Core pass only (skip enrichment)
python -m diaremot.cli core audio/sample.mp3 --outdir outputs/core

# Enrich existing core pass
python -m diaremot.cli enrich audio/sample.mp3 --outdir outputs/core
```

**Programmatic usage:**
```python
from diaremot.pipeline.audio_pipeline_core import run_pipeline, build_pipeline_config

# Configure
config = build_pipeline_config({
    "whisper_model": "faster-whisper-tiny.en",
    "compute_type": "int8",
    "vad_threshold": 0.35,
    "enable_sed": True,
    "disable_affect": False,
})

# Run
result = run_pipeline("audio.wav", "outputs/", config=config)

# Access results
print(f"Speakers: {result['num_speakers']}")
print(f"Segments: {result['num_segments']}")
print(f"Output: {result['out_dir']}")
```

### 5. Development Workflow

**Typical workflow:**

1. **Make changes** in appropriate module
   ```bash
   # Edit src/diaremot/pipeline/stages/my_stage.py
   ```

2. **Run linter**
   ```bash
   ruff check src/diaremot/pipeline/stages/my_stage.py
   ruff format src/diaremot/pipeline/stages/my_stage.py
   ```

3. **Run type checker** (optional)
   ```bash
   mypy src/diaremot/pipeline/stages/my_stage.py
   ```

4. **Run tests**
   ```bash
   # Specific test
   pytest tests/pipeline/test_my_stage.py -v

   # All tests
   pytest tests/ -v

   # With coverage
   pytest tests/ --cov=diaremot --cov-report=html
   ```

5. **Verify schema unchanged** (if applicable)
   ```bash
   python -c "from diaremot.pipeline.outputs import SEGMENT_COLUMNS; assert len(SEGMENT_COLUMNS) == 53"
   ```

6. **Integration test on sample audio**
   ```bash
   python -m diaremot.cli run -i data/sample.mp3 -o /tmp/test_output
   ```

7. **Check outputs**
   ```bash
   # Verify CSV structure
   head /tmp/test_output/diarized_transcript_with_emotion.csv

   # Verify column count
   python -c "import pandas as pd; df = pd.read_csv('/tmp/test_output/diarized_transcript_with_emotion.csv'); print(f'Columns: {len(df.columns)}')"
   ```

### 6. Working with the Web API

**Installing web dependencies:**
```bash
# Install with web extras
pip install -e ".[web]"

# Verify installation
python -c "from diaremot.web.api.app import app; print('✓ Web API ready')"
```

**Running the development server:**
```bash
# Option 1: Helper script
python src/diaremot/web/server.py

# Option 2: Uvicorn directly with hot reload
uvicorn diaremot.web.api.app:app --reload --port 8000

# Option 3: Custom port
uvicorn diaremot.web.api.app:app --reload --port 9000
```

**Testing the API:**
```bash
# Run the provided test script
python test_web_api.py

# Or test manually with curl
curl http://localhost:8000/health
curl http://localhost:8000/config/schema
```

**Frontend development:**
```bash
# Install dependencies
cd frontend/frontend
npm install

# Start development server
npm run dev

# Build for production
npm run build
npm start
```

**Common API patterns:**

1. **Adding a new endpoint:**
   ```python
   # In src/diaremot/web/api/app.py or api/routes/
   @app.post("/api/my-endpoint")
   async def my_endpoint(request: MyRequestModel):
       """New API endpoint."""
       # Implementation
       return {"status": "success"}
   ```

2. **Adding WebSocket functionality:**
   ```python
   # In src/diaremot/web/api/websocket.py
   @app.websocket("/ws/my-stream")
   async def my_stream(websocket: WebSocket):
       await websocket.accept()
       # Stream data to client
       await websocket.send_json({"progress": 50})
   ```

3. **Updating configuration schema:**
   ```python
   # In src/diaremot/web/config_schema.py
   # Add new parameter to PARAMETER_SCHEMA
   ```

### 7. Debugging Pipeline Issues

**Enable debug logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

from diaremot.pipeline.audio_pipeline_core import run_pipeline

result = run_pipeline("audio.wav", "outputs/")
```

**Run diagnostics:**
```bash
# Check dependencies
python -m diaremot.cli diagnostics

# Strict version check
python -m diaremot.cli diagnostics --strict
```

**Verify models:**
```bash
# Check model directory
echo $DIAREMOT_MODEL_DIR
ls -lh $DIAREMOT_MODEL_DIR/

# Verify critical models
python -c "
from pathlib import Path
import os
models = [
    'Diarization/silero_vad/silero_vad.onnx',
    'Diarization/ecapa-onnx/ecapa_tdnn.onnx',
    'Affect/ser8/model.int8.onnx',
]
model_dir = Path(os.getenv('DIAREMOT_MODEL_DIR', 'D:/models'))
missing = [m for m in models if not (model_dir / m).exists()]
print('✓ All models present' if not missing else f'Missing: {missing}')
"
```

**Clear caches:**
```bash
# Clear all caches
rm -rf .cache/

# Clear specific cache
rm -rf .cache/hf/
rm -rf .cache/torch/

# Use CLI flag
python -m diaremot.cli run -i audio.wav -o outputs/ --clear-cache
```

---

## Critical Constraints

### Inviolable Rules

1. **CPU-only** - No GPU code, no CUDA dependencies
2. **Schema stability** - `SEGMENT_COLUMNS` order is contractual (currently 53 columns)
3. **Stage count** - Exactly 11 stages in `PIPELINE_STAGES`
4. **Entry points** - Don't rename `cli.py::app`, `run_pipeline()`
5. **Python version** - 3.11-3.12 only (3.13+ breaks dependencies)
6. **ONNX preferred** - Always try ONNX before PyTorch fallback
7. **Paralinguistics required** - Stage cannot fail silently

### Performance Guardrails

- **Max ASR threads:** 1 (CTranslate2 limitation)
- **Max OpenMP threads:** 4 (diminishing returns above)
- **ASR window size:** 480 seconds (8 minutes)
- **Auto-chunk threshold:** 30 minutes
- **Affect window:** 30 seconds
- **Batch target size:** 60 seconds (transcription)
- **Max batch duration:** 300 seconds (transcription)

### File Path Conventions

- **Models:** `${DIAREMOT_MODEL_DIR}/*.onnx`
- **Speaker registry:** Default `speaker_registry.json` (configurable)
- **Cache:** `.cache/hf/`, `.cache/transformers/`, `.cache/torch/`
- **Checkpoints:** `checkpoints/{session_id}/`
- **Outputs:** `<outdir>/diarized_transcript_with_emotion.csv`, etc.

---

## Troubleshooting Guide

### Common Issues

#### "Module not found" errors
```bash
# Ensure package installed
pip install -e .

# Verify imports
python -c "import diaremot; print(diaremot.__file__)"
```

#### "Model not found" errors
```bash
# Check model directory
echo $DIAREMOT_MODEL_DIR
ls -lh $DIAREMOT_MODEL_DIR/

# Verify critical models
python scripts/audit_models_all.py
```

#### Poor diarization results
```bash
# Try adjusting VAD threshold (lower = more sensitive)
diaremot run -i audio.wav -o out/ --vad-threshold 0.25

# Increase AHC distance threshold for fewer speakers
diaremot run -i audio.wav -o out/ --ahc-distance-threshold 0.20

# Add more speech padding
diaremot run -i audio.wav -o out/ --vad-speech-pad-sec 0.30
```

#### Slow processing
```bash
# Use int8 quantization
diaremot run -i audio.wav -o out/ --asr-compute-type int8

# Disable optional stages
diaremot run -i audio.wav -o out/ --disable-sed --disable-affect

# Use fast profile
diaremot run -i audio.wav -o out/ --profile fast
```

#### Memory issues with long files
```bash
# Auto-chunking activates at 30 minutes
# Force smaller chunks:
diaremot run -i long_audio.wav -o out/ \
    --chunk-threshold-minutes 15.0 \
    --chunk-size-minutes 10.0 \
    --chunk-overlap-seconds 20.0
```

### Common Confusion

**Q: Where is the `auto_tune` stage?**
A: There isn't one in `PIPELINE_STAGES`. VAD tuning happens inline in orchestrator initialization.

**Q: Why does orchestrator override VAD threshold?**
A: Adaptive tuning based on audio energy. User CLI flags still take precedence.

**Q: What's the difference between `norm_tx` and `segments_final`?**
A: `norm_tx` has ASR output only (7 fields). `segments_final` adds affect, paralinguistics, SED (53 fields total).

**Q: How does caching work?**
A: Four primary artefacts per audio: `preprocessed_audio.npy` + `preprocessed.meta.json` (audio + validation keys), `diar.json` (turns), and `tx.json` (transcripts). Legacy `preprocessed.npz` is still read for backward compatibility.

**Q: Why are column counts different in docs?**
A: Schema evolved. Current version has **53 columns** (verify in `outputs.py::SEGMENT_COLUMNS`).

---

## Quick Reference

### Entry Points

**CLI:**
```bash
python -m diaremot.cli <command> [options]
# or
diaremot <command> [options]
```

**Programmatic:**
```python
from diaremot.pipeline.audio_pipeline_core import run_pipeline
result = run_pipeline("audio.wav", "outputs/")
```

**Web API:**
```bash
# Development server
python src/diaremot/web/server.py

# Or using uvicorn directly
uvicorn diaremot.web.api.app:app --reload --port 8000

# Production mode
uvicorn diaremot.web.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

**API (Programmatic):**
```python
from diaremot.web.api.app import app  # FastAPI app
```

### Key Files

| File | Purpose |
|------|---------|
| `src/diaremot/cli.py` | CLI entry point (Typer app) |
| `src/diaremot/pipeline/orchestrator.py` | Main pipeline class |
| `src/diaremot/pipeline/stages/__init__.py` | PIPELINE_STAGES registry |
| `src/diaremot/pipeline/stages/base.py` | PipelineState definition |
| `src/diaremot/pipeline/outputs.py` | SEGMENT_COLUMNS (53 cols) |
| `src/diaremot/pipeline/config.py` | PipelineConfig schema |
| `src/diaremot/pipeline/errors.py` | Structured exceptions |
| `src/diaremot/web/api/app.py` | FastAPI web application |
| `src/diaremot/web/config_schema.py` | Web UI configuration schema |
| `pyproject.toml` | Package configuration |
| `requirements.txt` | Python dependencies |

### Essential Commands

```bash
# Run pipeline
diaremot run -i audio.wav -o outputs/

# Resume
diaremot resume -i audio.wav -o outputs/

# Smoke test
diaremot smoke --outdir /tmp/test

# Diagnostics
diaremot diagnostics

# Web API (requires pip install -e ".[web]")
python src/diaremot/web/server.py
uvicorn diaremot.web.api.app:app --reload

# Frontend (requires npm install in frontend/frontend)
cd frontend/frontend && npm run dev

# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/

# Type check
mypy src/
```

### Environment Variables

**Required:**
```bash
export DIAREMOT_MODEL_DIR=/path/to/models
export HF_HOME=./.cache
export TRANSFORMERS_CACHE=./.cache/transformers
export TORCH_HOME=./.cache/torch
```

**Performance:**
```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_MAX_THREADS=4
export TOKENIZERS_PARALLELISM=false
```

**Optional model overrides:**
```bash
export SILERO_VAD_ONNX_PATH=/path/to/silero_vad.onnx
export ECAPA_ONNX_PATH=/path/to/ecapa_tdnn.onnx
export DIAREMOT_SER_ONNX=/path/to/ser8/model.int8.onnx
```

### Stage Input → Output Summary

| Stage | Input | Output |
|-------|-------|--------|
| 1. dependency_check | None | Environment validation |
| 2. preprocess | `input_audio_path` | `y, sr, duration_s, health, audio_sha16` |
| 3. background_sed | `y, sr` | `sed_info` |
| 4. diarize | `y, sr` | `turns` |
| 5. transcribe | `turns, y, sr` | `norm_tx` |
| 6. paralinguistics | `norm_tx, y, sr` | `para_map` |
| 7. affect_and_assemble | `norm_tx, para_map, sed_info, y, sr` | `segments_final` (53 cols) |
| 8. overlap_interruptions | `segments_final` | `overlap_stats, per_speaker_interrupts` |
| 9. conversation_analysis | `segments_final, overlap_stats` | `conv_metrics` |
| 10. speaker_rollups | `segments_final, interrupts` | `speakers_summary` |
| 11. outputs | All state | CSV/JSONL/HTML/PDF files |

### Output Files

**Primary:**
- `diarized_transcript_with_emotion.csv` - 53 columns, all segment data
- `segments.jsonl` - Same data, JSONL format
- `summary.html` - Interactive HTML report

**Supporting:**
- `timeline.csv` - Simplified timeline
- `diarized_transcript_readable.txt` - Human-friendly text
- `speakers_summary.csv` - Per-speaker aggregates
- `qc_report.json` - Processing diagnostics
- `events_timeline.csv` - Sound event timeline (if SED ran)
- `speaker_registry.json` - Persistent speaker embeddings
- `conversation_metrics.csv` - Turn-taking metrics
- `overlap_summary.csv` - Overlap statistics
- `interruptions_by_speaker.csv` - Per-speaker interruptions
- `audio_health.csv` - Preprocessing QA metrics

---

## Document Maintenance

**Update this file when:**
- Adding/removing pipeline stages
- Changing critical file paths
- Modifying CSV schema
- Updating dependencies with breaking changes
- Changing development workflows
- Adding new design patterns
- Modifying architecture

**Last reviewed:** 2025-11-22
**Review frequency:** Every major version bump
**Schema version:** v3 (53 columns)
**Pipeline stages:** 11
**Major updates since v2.2.0:** Web API + Next.js frontend added

---

## Additional Resources

- [README.md](README.md) - Complete user guide (1200 lines)
- [DATAFLOW.md](DATAFLOW.md) - Detailed pipeline data flow (600+ lines)
- [WEB_API_README.md](WEB_API_README.md) - Web API installation and usage guide
- [WEB_APP_PROGRESS.md](WEB_APP_PROGRESS.md) - Frontend development progress
- [GEMINI.md](GEMINI.md) - General AI assistant context
- [AGENTS.md](AGENTS.md) - Autonomous agent setup
- [MODEL_MAP.md](MODEL_MAP.md) - Model inventory and paths
- [CLOUD_BUILD_GUIDE.md](CLOUD_BUILD_GUIDE.md) - Cloud deployment
- [docs/pipeline_stage_analysis.md](docs/pipeline_stage_analysis.md) - Stage responsibilities

---

**End of CLAUDE.md**
