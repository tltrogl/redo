# DiaRemot Setup Script for Windows
# This script sets up the Python environment and installs all dependencies

param(
    [switch]$Force,
    [switch]$RuntimeOnly,
    [switch]$SkipDiagnostics
)

$ErrorActionPreference = "Stop"

$PYTHON_VERSION = "3.11"
$ALT_PYTHON_VERSION = "3.12"
$VENV_DIR = ".venv"
$PROJECT_ROOT = $PSScriptRoot

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "DiaRemot Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Function to write colored output
function Write-Success { param([string]$Message) Write-Host $Message -ForegroundColor Green }
function Write-Info    { param([string]$Message) Write-Host $Message -ForegroundColor Cyan }
function Write-Warning-Custom { param([string]$Message) Write-Host $Message -ForegroundColor Yellow }
function Write-Error-Custom   { param([string]$Message) Write-Host $Message -ForegroundColor Red }

function Confirm-Action {
    param(
        [string]$Message,
        [ValidateSet('Yes','No')]
        [string]$Default = 'No'
    )

    $promptSuffix = if ($Default -eq 'Yes') { '(Y/n)' } else { '(y/N)' }

    if ($Force.IsPresent) {
        Write-Warning-Custom "$Message $promptSuffix -- proceeding automatically due to -Force."
        return $true
    }

    if (-not $script:IsInteractive) {
        Write-Warning-Custom "$Message $promptSuffix (no interactive input detected; defaulting to $Default)."
        return ($Default -eq 'Yes')
    }

    $response = Read-Host "$Message $promptSuffix"
    if ([string]::IsNullOrWhiteSpace($response)) {
        $response = $Default.Substring(0, 1)
    }

    return $response -match '^[Yy]'
}

$script:IsInteractive = $true
try {
    if ([Console]::IsInputRedirected -or [Console]::IsOutputRedirected -or [Console]::IsErrorRedirected) {
        $script:IsInteractive = $false
    }
}
catch {
    if ($Host.Name -ne 'ConsoleHost') {
        $script:IsInteractive = $false
    }
}

if ($Force.IsPresent) {
    Write-Warning-Custom "-Force specified; all prompts will be auto-confirmed."
}
elseif (-not $script:IsInteractive) {
    Write-Warning-Custom "No interactive console detected; default responses will be used. Re-run with -Force to auto-confirm prompts."
}

if ($RuntimeOnly.IsPresent) {
    Write-Info "Runtime-only mode enabled: developer extras will be skipped."
}
if ($SkipDiagnostics.IsPresent) {
    Write-Info "Diagnostics will be skipped per -SkipDiagnostics."
}

# Check for Python
Write-Host "Checking for Python ${PYTHON_VERSION}/${ALT_PYTHON_VERSION}..."
$pythonCmd = $null

# Try different Python commands
$pythonCommands = @("py -$PYTHON_VERSION", "py -$ALT_PYTHON_VERSION", "python$PYTHON_VERSION", "python$ALT_PYTHON_VERSION", "python3", "python")

foreach ($cmd in $pythonCommands) {
    try {
        $versionOutput = & $cmd.Split() --version 2>&1 | Out-String
        if ($versionOutput -match "Python (3\.1[12]\.\d+)") {
            $pythonCmd = $cmd
            $foundVersion = $matches[1]
            Write-Success "Found: Python $foundVersion"
            break
        }
    }
    catch {
        continue
    }
}

if (-not $pythonCmd) {
    Write-Error-Custom "Error: Python 3.11 or 3.12 not found!"
    Write-Host "Please install Python 3.11 or 3.12 from https://www.python.org/downloads/"
    Write-Host "Make sure to check 'Add Python to PATH' during installation"
    exit 1
}
Write-Host ""

# Check for FFmpeg
Write-Host "Checking for FFmpeg..."
try {
    $ffmpegVersion = & ffmpeg -version 2>&1 | Select-Object -First 1
    if ($ffmpegVersion -match "ffmpeg version (\S+)") {
        Write-Success "Found: FFmpeg $($matches[1])"
    }
}
catch {
    Write-Warning-Custom "Warning: FFmpeg not found!"
    Write-Host "FFmpeg is required for audio processing. Install with:"
    Write-Host "  1. Download from https://ffmpeg.org/download.html"
    Write-Host "  2. Or use: winget install ffmpeg"
    Write-Host "  3. Or use: choco install ffmpeg"
    Write-Host ""
    if (-not (Confirm-Action "Continue anyway?" "No")) {
        exit 1
    }
}
Write-Host ""

# Create virtual environment
if (Test-Path $VENV_DIR) {
    Write-Warning-Custom "Virtual environment already exists at $VENV_DIR"
    if (Confirm-Action "Remove and recreate?" "No") {
        Write-Host "Removing existing virtual environment..."
        Remove-Item -Recurse -Force $VENV_DIR
    }
    else {
        Write-Host "Using existing virtual environment..."
    }
}

if (-not (Test-Path $VENV_DIR)) {
    Write-Host "Creating virtual environment..."
    & $pythonCmd.Split() -m venv $VENV_DIR
    Write-Success "Virtual environment created"
}
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..."
$activateScript = Join-Path $VENV_DIR "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Success "Virtual environment activated"
}
else {
    Write-Error-Custom "Error: Activation script not found at $activateScript"
    exit 1
}
Write-Host ""

# Upgrade pip, wheel, setuptools
Write-Host "Upgrading pip, wheel, and setuptools..."
python -m pip install --upgrade pip wheel setuptools | Out-Null
Write-Success "Package managers upgraded"
Write-Host ""

# Install project dependencies
Write-Host "Installing project dependencies..."
if (Test-Path (Join-Path $PROJECT_ROOT "requirements.txt")) {
    python -m pip install -r (Join-Path $PROJECT_ROOT "requirements.txt") | Out-Null
    Write-Success "Dependencies installed"
}
else {
    Write-Error-Custom "Error: requirements.txt not found!"
    exit 1
}
Write-Host ""

# Install package in editable mode
Write-Host "Installing diaremot package in editable mode..."
function Install-Editable {
    param(
        [switch]$RuntimeOnly
    )

    if ($RuntimeOnly) {
        python -m pip install -e $PROJECT_ROOT | Out-Null
        return
    }

    try {
        python -m pip install -e "$PROJECT_ROOT[dev]" | Out-Null
    }
    catch {
        Write-Warning-Custom "Editable install with dev extras failed; falling back to runtime-only install."
        python -m pip install -e $PROJECT_ROOT | Out-Null
    }
}

Install-Editable -RuntimeOnly:$RuntimeOnly
Write-Success "Package installed"
Write-Host ""

# Create necessary directories
Write-Host "Creating necessary directories..."
$directories = @(
    ".cache\hf",
    ".cache\hub",
    ".cache\torch",
    ".cache\transformers",
    "checkpoints",
    "outputs",
    "logs",
    "data"
)
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Success "Directories created"
Write-Host ""

# Check model directory
Write-Host "Checking model directory..."
$modelDir = $env:DIAREMOT_MODEL_DIR
if (-not $modelDir) {
    if (Test-Path "D:\models") {
        $modelDir = "D:\models"
    }
    elseif (Test-Path ".\models") {
        $modelDir = ".\models"
    }
    elseif (Test-Path "$HOME\models") {
        $modelDir = "$HOME\models"
    }
    else {
        Write-Warning-Custom "Warning: No model directory found"
        Write-Host "Set DIAREMOT_MODEL_DIR environment variable or create:"
        Write-Host "  D:\models (recommended)"
        Write-Host "  .\models"
        Write-Host "  ~\models"
        $modelDir = ""
    }
}

if ($modelDir) {
    Write-Success "Model directory: $modelDir"
}
else {
    Write-Warning-Custom "No model directory configured"
}
Write-Host ""

# Run diagnostics
if ($SkipDiagnostics.IsPresent) {
    Write-Info "Skipping diagnostics per -SkipDiagnostics."
}
else {
    Write-Host "Running diagnostics..."
    try {
        python -m diaremot.cli diagnostics
        Write-Success "Diagnostics completed"
    }
    catch {
        Write-Warning-Custom "Diagnostics exited with $($_.Exception.Message). Review the output above for details."
    }
    Write-Host ""
}

# Setup complete
Write-Host "========================================" -ForegroundColor Cyan
Write-Success "Setup Complete!"
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the environment in future sessions:"
Write-Host "  .\$VENV_DIR\Scripts\Activate.ps1"
Write-Host ""
Write-Host "To run the pipeline:"
Write-Host "  python -m diaremot.cli run --input audio.wav --outdir outputs\"
Write-Host ""
Write-Host "To run smoke test:"
Write-Host "  python -m diaremot.cli smoke --outdir outputs\"
Write-Host ""
Write-Host "Environment variables to set (add to PowerShell profile or system environment):"
if ($modelDir) {
    Write-Host "  `$env:DIAREMOT_MODEL_DIR = `"$modelDir`""
}
else {
    Write-Host "  `$env:DIAREMOT_MODEL_DIR = `"D:\models`""
}
Write-Host "  `$env:HF_HOME = `".\.cache`""
Write-Host "  `$env:HUGGINGFACE_HUB_CACHE = `".\.cache\hub`""
Write-Host "  `$env:TRANSFORMERS_CACHE = `".\.cache\transformers`""
Write-Host "  `$env:TORCH_HOME = `".\.cache\torch`""
Write-Host "  `$env:OMP_NUM_THREADS = `"4`""
Write-Host "  `$env:MKL_NUM_THREADS = `"4`""
Write-Host "  `$env:NUMEXPR_MAX_THREADS = `"4`""
Write-Host "  `$env:TOKENIZERS_PARALLELISM = `"false`""
Write-Host ""
Write-Host "To add these permanently, run (as Administrator):"
Write-Host "  [System.Environment]::SetEnvironmentVariable('DIAREMOT_MODEL_DIR', '$($modelDir -replace '\\','\\')', 'User')"
Write-Host ""
