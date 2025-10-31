#!/usr/bin/env bash
# Helper to run DiaRemot with sensible CPU/thread defaults on the VM.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage: scripts/diaremot_run.sh [run|resume|smoke] [OPTIONS] [ARGS]

Options recognised by this wrapper:
  --threads N        Override the thread count used for ASR/ONNX workloads.
  -h, --help         Show this help message.

All remaining arguments are passed straight to `python -m diaremot.cli <command>`.
If no command is supplied, `run` is used. For the run command, you can pass an
audio file directly (e.g. `audio/troglin.M4A`) and the CLI will write results to
<input parent>/outs/<input stem> by default.

Environment variables honoured:
  DIAREMOT_THREADS   Alternate shorthand for --threads.
  DIAREMOT_MODEL_DIR Primary model directory (defaults to <repo>/models if unset).
USAGE
}

# Parse wrapper-specific options.
THREAD_OVERRIDE="${DIAREMOT_THREADS:-}"
COMMAND="run"
POSITIONAL=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    run|resume|smoke|diagnostics)
      COMMAND="$1"
      shift
      ;;
    --threads)
      [[ $# -ge 2 ]] || { echo "Error: --threads requires a value." >&2; exit 1; }
      THREAD_OVERRIDE="$2"
      shift 2
      ;;
    --threads=*)
      THREAD_OVERRIDE="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

set -- "${POSITIONAL[@]}"

# Determine logical CPU count if threads not explicitly set.
if [[ -n "${THREAD_OVERRIDE}" ]]; then
  THREADS="${THREAD_OVERRIDE}"
else
  THREADS="$(nproc --all 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
fi

# Activate the local virtual environment (.balls) expected on the VM.
VENV_ACTIVATE="${REPO_ROOT}/.balls/bin/activate"
if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "Virtualenv not found at ${VENV_ACTIVATE}. Activate your environment first." >&2
  exit 1
fi
source "${VENV_ACTIVATE}"

# Export thread-related environment variables if not already set by the user.
export CT2_NUM_THREADS="${CT2_NUM_THREADS:-${THREADS}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${THREADS}}"

# Ensure a default model root is present.
if [[ -z "${DIAREMOT_MODEL_DIR:-}" ]]; then
  export DIAREMOT_MODEL_DIR="${REPO_ROOT}/models"
fi

# Only add --model-root if the caller did not supply one explicitly.
MODEL_ROOT_ARGS=()
MODEL_FLAG_PRESENT="false"
for arg in "$@"; do
  case "$arg" in
    --model-root|--model-root=*)
      MODEL_FLAG_PRESENT="true"
      break
      ;;
  esac
done

if [[ "${MODEL_FLAG_PRESENT}" == "false" ]]; then
  MODEL_ROOT_ARGS+=(--model-root "${DIAREMOT_MODEL_DIR}")
fi

echo "Running DiaRemot (${COMMAND}) with ${THREADS} threads (CT2/OMP/MKL)." >&2
python -m diaremot.cli "${COMMAND}" "${MODEL_ROOT_ARGS[@]}" "$@"

