<#
.SYNOPSIS
    Batch-run DiaRemot over a directory of video/audio files and record a CSV manifest.

.DESCRIPTION
    This wrapper mirrors the loop you ran earlier. For each media file matching the specified filter
    it creates an output folder under the target root named after the input file stem and runs
    `python -m diaremot.cli <mode> <input> --outdir <out>`.

    A CSV log is appended (or created) at the path specified by -LogFile.

.PARAMETER BaseDir
    Directory containing media files to process (default: current directory).

.PARAMETER TargetRoot
    Root directory where per-file outputs will be created (default: ./transcribe).

.PARAMETER PythonExe
    Path to the python executable to use (default: 'python').

.PARAMETER Mode
    DiaRemot CLI subcommand to run: 'core' or 'run' (default: 'core').

.PARAMETER Pattern
    File filter pattern (default: '*.mp4').

.PARAMETER ModelRoot
    Optional --model-root path to pass to the CLI.

.PARAMETER DryRun
    If present, commands are printed but not executed.

.PARAMETER LogFile
    CSV log file path (default: '<TargetRoot>/run_manifest.csv').
#>

param(
    [string]$BaseDir = (Get-Location).ProviderPath,
    [string]$TargetRoot = "./transcribe",
    [string]$PythonExe = "python",
    [ValidateSet('core','run')]
    [string]$Mode = "core",
    [string]$Pattern = "*.mp4",
    [string]$ModelRoot = "",
    [switch]$DryRun = $false,
    [string]$LogFile = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Normalize paths
$BaseDir = (Resolve-Path $BaseDir).Path
$TargetRoot = (Resolve-Path (Join-Path $TargetRoot "" -Resolve) -ErrorAction SilentlyContinue).ProviderPath 2>$null
if (-not $TargetRoot) { $TargetRoot = (Resolve-Path (New-Item -ItemType Directory -Path $TargetRoot -Force).FullName).ProviderPath }

if ([string]::IsNullOrEmpty($LogFile)) { $LogFile = Join-Path $TargetRoot 'run_manifest.csv' }

# Ensure target root exists
New-Item -ItemType Directory -Path $TargetRoot -Force | Out-Null

# Prepare log header if missing
if (-not (Test-Path $LogFile)) {
    "timestamp,input,output,status,started,ended,exit_code" | Out-File -FilePath $LogFile -Encoding utf8
}

# Find files
$files = Get-ChildItem -Path $BaseDir -Filter $Pattern -File | Sort-Object Name
if (-not $files) {
    Write-Host "No files matching '$Pattern' found in $BaseDir"
    exit 0
}

foreach ($f in $files) {
    $stem = $f.BaseName -replace '[\\/:]', '-'
    $out = Join-Path $TargetRoot $stem
    New-Item -ItemType Directory -Path $out -Force | Out-Null

    $args = @()
    if ($Mode -eq 'core') {
        $args += $f.FullName
        $args += '--outdir'
        $args += $out
    } else {
        $args += '-m'
        $args += 'diaremot.cli'
        $args += 'run'
        $args += '--input'
        $args += $f.FullName
        $args += '--outdir'
        $args += $out
    }

    if (-not [string]::IsNullOrEmpty($ModelRoot)) {
        $args += '--model-root'
        $args += $ModelRoot
    }

    $cmdDisplay = if ($Mode -eq 'core') { "$PythonExe -m diaremot.cli core $($f.FullName) --outdir $out" } else { "$PythonExe -m diaremot.cli run --input $($f.FullName) --outdir $out" }

    $start = Get-Date
    if ($DryRun) {
        Write-Host "[DryRun] $cmdDisplay"
        $status = 'DryRun'
        $exitCode = -1
        $end = Get-Date
    }
    else {
        Write-Host "Running: $cmdDisplay"
        try {
            # Use Start-Process to stream output to console
            $proc = Start-Process -FilePath $PythonExe -ArgumentList $args -NoNewWindow -Wait -PassThru -RedirectStandardOutput ([Console]::Out) -RedirectStandardError ([Console]::Out)
            $exitCode = $proc.ExitCode
            $status = if ($exitCode -eq 0) { 'Success' } else { 'Failed' }
        }
        catch {
            $exitCode = 1
            $status = 'Failed'
            Write-Host "Error running pipeline for $($f.FullName): $_" -ForegroundColor Yellow
        }
        $end = Get-Date
    }

    # Append CSV log
    $timestamp = (Get-Date).ToString('o')
    $csvLine = "{0},{1},{2},{3},{4},{5},{6}" -f $timestamp, ($f.FullName -replace ',', '`,'), ($out -replace ',', '`,'), $status, $($start.ToString('o')), $($end.ToString('o')), $exitCode
    Add-Content -Path $LogFile -Value $csvLine -Encoding utf8
}

Write-Host "Batch run complete. Manifest: $LogFile"
