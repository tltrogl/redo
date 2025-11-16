#requires -Version 5.1
<#
NAME
    process_evidence_videos.ps1

SYNOPSIS
    Batch-process MP4 files with DiaRemot (transcription-only), keeping a shared speaker registry
    within the evidence folder and logging run metadata.

USAGE
    # Dry run (shows commands and records DryRun entries in the log):
    pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\process_evidence_videos.ps1 -DryRun

    # Full batch run (default):
    pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\process_evidence_videos.ps1

PARAMETERS
    -VideoDir   : Top-level directory with videos (default: D:/court/disc/evidence/video)
    -OutputsSubDir : Where to place per-video outputs (default: 'outputs' under VideoDir)
    -RegistrySubDir : Subfolder for shared registry (default: '.registry' under VideoDir)
    -ModelRoot  : Local model folder (default: './models')
    -LogFileName: Name of the metadata log file under OutputsSubDir (default: 'processing.log')
    -Single     : Path to a single video file to process (optional)
    -DryRun     : Show intended commands and append DryRun entries to log
#>

param(
    [string]$VideoDir = "D:/court/disc/evidence/video",
    [string]$OutputsSubDir = "outputs",
    [string]$RegistrySubDir = ".registry",
    [string]$ModelRoot = "./models",
    [string]$LogFileName = "processing.log",
    [string]$Single = "",
    [switch]$DryRun
)

function _resolve-path-trust($p) {
    try { return (Resolve-Path $p -ErrorAction Stop).Path } catch { return $p }
}

$VideoDir = _resolve-path-trust $VideoDir
$RegistryDir = Join-Path $VideoDir $RegistrySubDir
$OutputsRoot = Join-Path $VideoDir $OutputsSubDir
$RegistryPath = Join-Path $RegistryDir "speaker_registry.json"
$LogPath = Join-Path $OutputsRoot $LogFileName

if (-not (Test-Path $VideoDir)) { Write-Host "Video directory not found: $VideoDir" -ForegroundColor Red; exit 1 }

New-Item -Path $RegistryDir -ItemType Directory -Force | Out-Null
New-Item -Path $OutputsRoot -ItemType Directory -Force | Out-Null

function Write-LogEntry {
    param(
        [string]$Video,
        [string]$Output,
        [string]$Status,
        [DateTime]$StartTime,
        [Nullable[DateTime]]$EndTime = $null,
        [int]$ExitCode = 0
    )

    $entry = [pscustomobject]@{
        timestamp = (Get-Date).ToString('o')
        video = $Video
        output = $Output
        registry_path = $RegistryPath
        status = $Status
        started = $StartTime.ToString('o')
        ended = if ($null -eq $EndTime) { $null } else { $EndTime.ToString('o') }
        exit_code = $ExitCode
    }
    $entry | ConvertTo-Json -Compress | Add-Content -Path $LogPath -Encoding utf8
}

function _run-one {
    param($videoPath)
    $videoOut = Join-Path $OutputsRoot (Split-Path $videoPath -LeafBase)
    New-Item -Path $videoOut -ItemType Directory -Force | Out-Null

    $pythonExe = "python"
    $envPython = Join-Path (Get-Location).Path '.balls\Scripts\python.exe'
    if (Test-Path $envPython) { $pythonExe = $envPython }

    $pythonArgs = @(
        '-m','diaremot.cli','run',
        '--input',"$videoPath",
        '--outdir',"$videoOut",
        '--model-root',"$ModelRoot",
        '--registry-path',"$RegistryPath",
        '--disable-affect','--disable-sed'
    )

    $start = Get-Date
    if ($DryRun) {
        Write-Host "Dry run: $pythonExe $($pythonArgs -join ' ')"
        Write-LogEntry -Video $videoPath -Output $videoOut -Status "DryRun" -StartTime $start
        return
    }

    Write-Host "Processing $videoPath -> $videoOut"
    & $pythonExe @pythonArgs
    $exit = $LASTEXITCODE
    $end = Get-Date
    $status = if ($exit -eq 0) { 'Success' } else { 'Failed' }
    Write-LogEntry -Video $videoPath -Output $videoOut -Status $status -StartTime $start -EndTime $end -ExitCode $exit
    if ($exit -ne 0) { Write-Host "Failed: $videoPath (exit $exit)" -ForegroundColor Yellow }
}

if ($Single -ne '') {
    if (-not (Test-Path $Single)) { Write-Host "Single file not found: $Single" -ForegroundColor Red; exit 1 }
    _run-one $Single
    exit 0
}

Get-ChildItem -Path $VideoDir -Filter *.mp4 -File | ForEach-Object {
    try { _run-one $_.FullName } catch { Write-Host "Error processing $($_.FullName): $_" -ForegroundColor Red }
}

Write-Host "Done. Log: $LogPath"
