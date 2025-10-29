#!/bin/bash
set -euo pipefail

# DiaRemot Setup Script for Linux/macOS
# This script provisions a virtual environment, installs runtime (and optionally
# development) dependencies, and runs basic diagnostics.

PYTHON_VERSION="3.11"
ALT_PYTHON_VERSION="3.12"
VENV_DIR=".venv"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colour codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m' # No Colour

ASSUME_YES=false
RUNTIME_ONLY=false
RUN_DIAGNOSTICS=true

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --yes, -y             Auto-confirm prompts (useful in CI)
  --runtime-only        Skip developer extras when installing diaremot
  --skip-diagnostics    Do not run python -m diaremot.cli diagnostics
  --help, -h            Show this message and exit
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --yes|-y)
            ASSUME_YES=true
            shift
            ;;
        --runtime-only)
            RUNTIME_ONLY=true
            shift
            ;;
        --skip-diagnostics)
            RUN_DIAGNOSTICS=false
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [ -t 0 ]; then
    HAS_TTY=true
else
    HAS_TTY=false
fi

if [ "$ASSUME_YES" = true ]; then
    echo -e "${YELLOW}--yes supplied; all prompts will be auto-confirmed.${NC}"
elif [ "$HAS_TTY" != true ]; then
    echo -e "${YELLOW}No interactive terminal detected; default responses will be used for prompts.${NC}"
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}DiaRemot Setup Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Helper to confirm actions with optional defaults
confirm() {
    local message="$1"
    local default="${2:-N}"
    local default_upper="${default^^}"
    local prompt_suffix

    if [[ "$default_upper" == "Y" ]]; then
        prompt_suffix="(Y/n)"
    else
        prompt_suffix="(y/N)"
    fi

    if [ "$ASSUME_YES" = true ]; then
        echo -e "${YELLOW}${message} ${prompt_suffix} -- proceeding automatically due to --yes.${NC}"
        return 0
    fi

    if [ "$HAS_TTY" != true ]; then
        echo -e "${YELLOW}${message} ${prompt_suffix} (no TTY detected; defaulting to ${default_upper}).${NC}"
        [[ "$default_upper" == "Y" ]]
        return $?
    fi

    local reply
    read -r -p "${message} ${prompt_suffix} " reply
    if [ -z "$reply" ]; then
        reply="$default_upper"
    else
        reply="${reply^^}"
    fi

    [[ "$reply" == "Y" ]]
}

info() {
    echo -e "${BLUE}$1${NC}"
}

success() {
    echo -e "${GREEN}$1${NC}"
}

warn() {
    echo -e "${YELLOW}$1${NC}"
}

error() {
    echo -e "${RED}$1${NC}" >&2
}

info "Checking for Python ${PYTHON_VERSION}/${ALT_PYTHON_VERSION}..."
PYTHON_CMD=""
if command -v python${PYTHON_VERSION} >/dev/null 2>&1; then
    PYTHON_CMD="python${PYTHON_VERSION}"
elif command -v python${ALT_PYTHON_VERSION} >/dev/null 2>&1; then
    PYTHON_CMD="python${ALT_PYTHON_VERSION}"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
    CURRENT_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    MAJOR_MINOR=${CURRENT_VERSION%.*}
    if [[ "$MAJOR_MINOR" != "3.11" && "$MAJOR_MINOR" != "3.12" ]]; then
        error "Error: Python 3.11 or 3.12 required, found ${CURRENT_VERSION}"
        exit 1
    fi
else
    error "Error: Python ${PYTHON_VERSION} not found!"
    echo "Please install Python 3.11 or 3.12 and try again."
    exit 1
fi
PYTHON_VERSION_FULL=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
success "Found: Python ${PYTHON_VERSION_FULL}"
echo ""

info "Checking for FFmpeg..."
if command -v ffmpeg >/dev/null 2>&1; then
    FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -n1 | awk '{print $3}')
    success "Found: FFmpeg ${FFMPEG_VERSION}"
else
    warn "FFmpeg not found!"
    echo "FFmpeg is required for audio processing. Install with:"
    echo "  Ubuntu/Debian: sudo apt-get install -y ffmpeg"
    echo "  macOS:         brew install ffmpeg"
    echo ""
    if ! confirm "Continue anyway?" "N"; then
        exit 1
    fi
fi
echo ""

if [ -d "$VENV_DIR" ]; then
    warn "Virtual environment already exists at ${VENV_DIR}"
    if confirm "Remove and recreate?" "N"; then
        info "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        info "Using existing virtual environment..."
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    success "Virtual environment created"
fi
echo ""

info "Activating virtual environment..."
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
success "Virtual environment activated"
echo ""

info "Upgrading pip, wheel, and setuptools..."
python -m pip install --upgrade pip wheel setuptools
success "Package managers upgraded"
echo ""

info "Installing project dependencies from requirements.txt..."
if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
    python -m pip install -r "$PROJECT_ROOT/requirements.txt"
    success "Runtime dependencies installed"
else
    error "Error: requirements.txt not found at $PROJECT_ROOT"
    exit 1
fi
echo ""

install_editable() {
    if [ "$RUNTIME_ONLY" = true ]; then
        info "Installing diaremot in editable mode (runtime dependencies only)..."
        python -m pip install -e "$PROJECT_ROOT"
        return
    fi

    info "Installing diaremot with development extras..."
    if python -m pip install -e "$PROJECT_ROOT[dev]"; then
        return
    fi

    warn "Editable install with dev extras failed; falling back to runtime-only editable install."
    python -m pip install -e "$PROJECT_ROOT"
}

install_editable
success "Package installation complete"
echo ""

info "Creating cache and output directories..."
mkdir -p .cache/hf .cache/torch .cache/transformers .cache/hub
mkdir -p checkpoints outputs logs data
success "Directories ensured"
echo ""

info "Checking model directory..."
MODEL_DIR=""
if [ -n "${DIAREMOT_MODEL_DIR:-}" ]; then
    MODEL_DIR="$DIAREMOT_MODEL_DIR"
elif [ -d "/srv/models" ]; then
    MODEL_DIR="/srv/models"
elif [ -d "$HOME/models" ]; then
    MODEL_DIR="$HOME/models"
elif [ -d "models" ]; then
    MODEL_DIR="models"
fi

if [ -n "$MODEL_DIR" ]; then
    success "Model directory: ${MODEL_DIR}"
else
    warn "No model directory configured"
    echo "Set DIAREMOT_MODEL_DIR or create one of the following:"
    echo "  /srv/models (recommended for shared hosts)"
    echo "  ~/models"
    echo "  ./models"
fi
echo ""

run_diagnostics() {
    if [ "$RUN_DIAGNOSTICS" != true ]; then
        info "Skipping diagnostics (--skip-diagnostics provided)."
        return
    fi

    info "Running diagnostics (python -m diaremot.cli diagnostics)..."
    set +e
    python -m diaremot.cli diagnostics
    status=$?
    set -e
    if [ $status -ne 0 ]; then
        warn "Diagnostics exited with status $status. Review the output above for details."
    else
        success "Diagnostics completed successfully"
    fi
    echo ""
}

run_diagnostics

echo -e "${BLUE}========================================${NC}"
success "Setup Complete!"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "To activate the environment in future sessions:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "To run the pipeline:"
echo "  python -m diaremot.cli run --input audio.wav --outdir outputs/"
echo ""
echo "To run smoke test:"
echo "  python -m diaremot.cli smoke --outdir outputs/"
echo ""
echo "Environment variables to set (add to ~/.bashrc or ~/.zshrc):"
echo "  export DIAREMOT_MODEL_DIR=${MODEL_DIR:-"/srv/models"}"
echo "  export HF_HOME=./.cache"
echo "  export HUGGINGFACE_HUB_CACHE=./.cache/hub"
echo "  export TRANSFORMERS_CACHE=./.cache/transformers"
echo "  export TORCH_HOME=./.cache/torch"
echo "  export OMP_NUM_THREADS=4"
echo "  export MKL_NUM_THREADS=4"
echo "  export NUMEXPR_MAX_THREADS=4"
echo "  export TOKENIZERS_PARALLELISM=false"
echo ""
