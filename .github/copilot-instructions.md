# DiaRemot - GitHub Copilot Instructions

## Project Overview

DiaRemot is a production-ready, CPU-only speech intelligence system (v2.2.0) that processes long-form audio (1-3 hours) into comprehensive diarized transcripts with deep affect, paralinguistic, and acoustic analysis. Built for research and production environments requiring detailed speaker analytics without GPU dependencies.

## Core Architecture

### 11-Stage Processing Pipeline
1. **dependency_check** – Validate runtime dependencies and model availability
2. **preprocess** – Audio resampling and loudness alignment with optional denoising plus auto-chunking for long files
3. **background_sed** – Sound event detection (music, keyboard, ambient noise)
4. **diarize** – Speaker segmentation with adaptive VAD tuning
5. **transcribe** – Speech-to-text with intelligent batching
6. **paralinguistics** – Voice quality and prosody extraction
7. **affect_and_assemble** – Emotion/intent analysis and segment assembly
8. **overlap_interruptions** – Turn-taking and interruption pattern analysis
9. **conversation_analysis** – Flow metrics and speaker dominance
10. **speaker_rollups** – Per-speaker statistical summaries
11. **outputs** – Generate CSV, JSON, HTML, PDF reports

### Key Technologies
- **Python 3.11 or 3.12** (3.13+ not yet supported)
- **ONNXRuntime** for all ML models (CPU-only, never use PyTorch fallbacks)
- **Faster-Whisper** (CTranslate2) for ASR
- **Silero VAD** + **ECAPA-TDNN** embeddings for diarization
- **PANNs CNN14** for sound event detection (527 AudioSet classes)
- **GoEmotions** for text emotion analysis (28 classes)
- **BART-MNLI** for intent classification

## Development Setup

### Prerequisites
- Python 3.11 or 3.12
- FFmpeg on PATH (`ffmpeg -version` must work)
- 4+ GB RAM
- 4+ CPU cores (recommended)

### Setup Commands

**Linux/macOS:**
```bash
# Run automated setup
./setup.sh

# Or manual setup:
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Windows:**
```powershell
# Run automated setup
.\setup.ps1

# Or manual setup:
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

**IMPORTANT:** Always activate the virtual environment (`.venv`) before running any commands.
- Linux/macOS: `source .venv/bin/activate`
- Windows: `.venv\Scripts\Activate.ps1`

### Models Setup
All required models are provided by `./assets/models.zip` (SHA-256: `33d0a9194de3cbd667fd329ab7c6ce832f4e1373ba0b1844ce0040a191abc483`). The setup scripts handle downloading and extracting models automatically.

Required model files:
- `./models/Diarization/ecapa-onnx/ecapa_tdnn.onnx`
- `./models/Diarization/silero_vad/silero_vad.onnx`
- `./models/Affect/ser8/model.int8.onnx`
- `./models/Affect/VAD_dim/model.onnx`
- `./models/Affect/sed_panns/model.onnx`
- `./models/text_emotions/model.int8.onnx`
- `./models/intent/model_int8.onnx`

## Building and Testing

### Linting
```bash
# Check code style and quality
ruff check src/ tests/

# Auto-fix issues where possible
ruff check --fix src/ tests/
```

### Type Checking
```bash
# Run mypy (non-zero exit is acceptable for informational warnings)
mypy src/ || true
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/diaremot --cov-report=html

# Run specific test file
pytest tests/test_diarization.py

# Run with verbose output
pytest -v
```

### Smoke Tests
```bash
# Full smoke test (all features enabled)
python -m diaremot.cli smoke --outdir ./smoke_output --enable-affect

# Core-only test (without affect enrichment)
python -m diaremot.cli smoke --outdir ./smoke_output
```

## Code Style and Conventions

### General Guidelines
- **Surgical edits**: Make the smallest possible changes across all affected files
- **No fallbacks**: Any fallback to alternate implementations counts as failure
- **Type hints**: Use type hints where they improve clarity
- **Docstrings**: Follow existing documentation patterns (Google-style)
- **Line length**: 120 characters (configured in pyproject.toml)
- **Import order**: Standard library, third-party, local imports (enforced by ruff)

### Python Conventions
- Use f-strings for string formatting
- Prefer pathlib.Path over os.path
- Use type hints for function signatures
- Follow PEP 8 naming conventions
- Use context managers for file operations
- Handle errors explicitly, don't silently suppress

### Architecture Patterns
- **Modular orchestration**: Core mixins live in `src/diaremot/pipeline/core/`
- **Structured errors**: Use `StageExecutionError` from `src/diaremot/pipeline/errors.py`
- **Façade pattern**: Components like `Transcriber` act as lightweight façades
- **Package organization**: Each major feature has its own package (affect, sed, intent, etc.)

## Repository Structure

```
src/diaremot/
├── cli.py                  # Main CLI entry point
├── pipeline/
│   ├── orchestrator.py    # 11-stage pipeline orchestrator
│   ├── stages/            # Stage implementations
│   ├── core/              # Core mixins
│   ├── transcription/     # ASR subsystem
│   ├── preprocess/        # Audio preprocessing
│   └── outputs.py         # Output schema (53 columns)
├── affect/                # Emotion and sentiment analysis
│   ├── paralinguistics/   # Voice quality and prosody
│   ├── vad_analyzer.py    # Valence-Arousal-Dominance
│   ├── speech_analyzer.py # SER-8 emotion recognition
│   └── text_analyzer.py   # GoEmotions text emotions
├── sed/                   # Sound event detection
├── intent/                # Intent classification
└── utils/                 # Shared utilities
tests/
├── affect/                # Affect subsystem tests
├── pipeline/              # Pipeline stage tests
├── transcription/         # ASR tests
└── conftest.py            # Shared test fixtures
```

## Key Files and Their Purpose

- **`src/diaremot/cli.py`** - CLI commands: `run`, `core`, `enrich`, `smoke`, `diagnostics`
- **`src/diaremot/pipeline/orchestrator.py`** - Main pipeline orchestrator with 11-stage execution
- **`src/diaremot/pipeline/outputs.py`** - Output schema definitions (53-column CSV)
- **`src/diaremot/affect/paralinguistics/extract.py`** - Voice quality and prosody extraction API
- **`src/diaremot/io/onnx_utils.py`** - ONNX model loading utilities
- **`pyproject.toml`** - Project configuration, dependencies, and tool settings
- **`requirements.txt`** - Pinned runtime dependencies
- **`setup.sh`** / **`setup.ps1`** - Platform-specific setup scripts

## Testing Guidelines

### What to Test
- Core pipeline stages (diarization, transcription, affect analysis)
- Edge cases (empty audio, single speaker, very long files)
- Error handling and recovery
- Output schema validation
- Model loading and caching

### Test Organization
- Use `conftest.py` for shared fixtures
- One test file per module/feature
- Group related tests in classes
- Use descriptive test names: `test_<feature>_<scenario>_<expected_outcome>`

### Mock External Dependencies
- Mock file I/O where appropriate
- Mock heavy models in unit tests (use real models in integration tests)
- Use fixtures for sample audio data

## Common Tasks

### Adding a New Pipeline Stage
1. Create stage implementation in `src/diaremot/pipeline/stages/`
2. Register stage in `src/diaremot/pipeline/stages/__init__.py`
3. Update orchestrator to call the new stage
4. Add tests in `tests/pipeline/`
5. Update DATAFLOW.md documentation

### Adding a New Model
1. Add model to `./models/` directory structure
2. Update `src/diaremot/utils/model_paths.py` with path resolution
3. Add ONNX loading logic in `src/diaremot/io/onnx_utils.py`
4. Update MODEL_MAP.md documentation
5. Update setup scripts to download model
6. Add model path to smoke test verification

### Modifying Output Schema
1. Update column definitions in `src/diaremot/pipeline/outputs.py`
2. Update CSV writer logic
3. Update tests to check new columns
4. Update documentation (README.md, DATAFLOW.md)
5. Increment version number in pyproject.toml

## Error Handling

### Structured Errors
Use `StageExecutionError` for pipeline failures:
```python
from diaremot.pipeline.errors import StageExecutionError

raise StageExecutionError(
    stage="stage_name",
    message="What went wrong",
    details={"key": "value"}
)
```

### Graceful Degradation
- Paralinguistics skips automatically when transcription fails
- PDF generation is optional (requires wkhtmltopdf)
- SED timeline is conditional on noise detection

### Never Skip Silently
- Always log warnings for degraded functionality
- Include failure information in QC report
- Surface errors in manifest output

## Output Files

### Primary Outputs
- **`diarized_transcript_with_emotion.csv`** - 53-column master transcript
- **`segments.jsonl`** - Full segment payloads with audio features
- **`summary.html`** - Interactive HTML report
- **`conversation_report.md`** - Narrative summary
- **`speakers_summary.csv`** - Per-speaker statistics
- **`qc_report.json`** - Quality control metrics

### Output Schema (53 columns)
Temporal, speaker, content, affect (V/A/D), emotion scores, voice quality (jitter, shimmer, HNR, CPPS), prosody (WPM, pause metrics, pitch), sound events, SNR estimates, quality flags, extended affect metadata (noise score, timeline events, ASR confidence/language/tokens, voice quality reliability).

## Performance Considerations

- Audio is automatically chunked for long files (1-3 hours)
- Intelligent batching for ASR (groups similar-length segments)
- Spectral magnitude caching to avoid redundant FFT operations
- Sweep-line algorithm for overlap detection (O(n log n))
- Shared STFT computation in preprocessing chain

## Security and Best Practices

### Never Commit
- Secrets or API keys
- Generated output files (`outputs_*`, `smoke_*`)
- Virtual environment directories (`.venv`)
- Cache directories (`.cache`, `__pycache__`)
- Build artifacts (`*.egg-info`, `dist/`, `build/`)

### Always Check
- Model file integrity (SHA-256 checksums)
- Input file format and codec support
- Available disk space for chunking
- FFmpeg availability before processing

## Documentation

### Core Documentation Files
- **README.md** - User guide and quick reference
- **DATAFLOW.md** - Detailed pipeline data flow
- **MODEL_MAP.md** - Complete model inventory
- **AGENTS.md** - Setup guide for agents (legacy)
- **agentscloud.md** - Cloud agent contract (strict mode)
- **GEMINI.md** - Project context for AI assistants

### When to Update Documentation
- Adding/removing pipeline stages
- Changing output schema
- Adding new dependencies
- Modifying setup procedures
- Changing model requirements

## CI/CD and Cloud Deployment

### Cloud Build
- Configuration: `cloudbuild.yaml`, `cloudbuild-cloudrun.yaml`
- Deployment scripts: `deploy-cloudrun.sh`, `deploy-cloudrun.ps1`
- Dockerfile for Cloud Run: `Dockerfile.cloudrun`

### Deployment Checklist
1. Lint and test locally
2. Update version in pyproject.toml
3. Update CHANGELOG or release notes
4. Verify model checksums
5. Test in clean environment
6. Deploy to staging first

## Agent-Specific Instructions

### For Autonomous Agents
See **agentscloud.md** for strict cloud execution contract:
- No mid-run approvals
- No fallbacks allowed
- Surgical but thorough edits
- Evidence-based reporting
- All required models from `./assets/models.zip`

### Execution Model
Plan → Execute → Self-Verify → Continue (no pauses for approval)

## Getting Help

### Diagnostic Commands
```bash
# Check environment and dependencies
python -m diaremot.cli diagnostics

# Verify model availability
ls -lh ./models/*/

# Check Python version
python --version

# Verify FFmpeg
ffmpeg -version
```

### Common Issues
- **Import errors**: Ensure virtual environment is activated
- **Model not found**: Run setup script to download models
- **FFmpeg not found**: Install FFmpeg and add to PATH
- **Out of memory**: Process shorter chunks or reduce parallelism
- **Slow processing**: Ensure running on CPU with 4+ cores

## Contact and Support

For questions or issues:
- Check existing GitHub issues
- Review DATAFLOW.md for pipeline details
- Consult MODEL_MAP.md for model information
- Read AGENTS.md for agent-specific guidance
