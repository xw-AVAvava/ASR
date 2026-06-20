[CmdletBinding()]
param(
    [switch]$UseCurrentEnvironment,
    [switch]$Cuda,
    [switch]$Pyannote,
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128",
    [switch]$SkipFfmpeg
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $Root ".venv"
$VenvPython = Join-Path $Venv "Scripts\python.exe"
$RuntimeRoot = Join-Path $Root ".runtime"

if ($UseCurrentEnvironment) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonCommand) {
        throw "Activate your Python environment before using -UseCurrentEnvironment."
    }
    $Python = $PythonCommand.Source
} else {
    if (-not (Test-Path $VenvPython)) {
        $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
        if ($PyLauncher) {
            & $PyLauncher.Source -3.12 -m venv $Venv
        } else {
            $SystemPython = Get-Command python -ErrorAction SilentlyContinue
            if (-not $SystemPython) {
                throw "Python 3.12 was not found. Install it or activate an existing environment."
            }
            & $SystemPython.Source -m venv $Venv
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create the virtual environment. Python 3.12 is recommended."
        }
    }
    $Python = $VenvPython
}

function Invoke-Python {
    param([string[]]$Arguments)
    & $script:Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $($Arguments -join ' ')"
    }
}

$env:PYTHONUTF8 = "1"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
Write-Host "Using Python: $Python"
Invoke-Python @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

if ($Cuda) {
    Write-Host "Installing CUDA 12.8 PyTorch from $TorchIndexUrl"
    Invoke-Python @(
        "-m", "pip", "install",
        "-r", (Join-Path $Root "requirements-cuda.txt"),
        "--extra-index-url", $TorchIndexUrl
    )
}

$Requirements = if ($Pyannote) {
    Join-Path $Root "requirements-pyannote.txt"
} else {
    Join-Path $Root "requirements.txt"
}
Invoke-Python @("-m", "pip", "install", "-r", $Requirements)

if ($Pyannote -and -not $SkipFfmpeg) {
    $SharedFfmpegRoot = Join-Path $RuntimeRoot "ffmpeg-shared"
    $FfmpegExe = Get-ChildItem -LiteralPath $SharedFfmpegRoot -Filter "ffmpeg.exe" -File -Recurse -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $FfmpegExe) {
        New-Item -ItemType Directory -Path $SharedFfmpegRoot -Force | Out-Null
        $Archive = Join-Path ([IO.Path]::GetTempPath()) "ffmpeg-master-latest-win64-gpl-shared.zip"
        $FfmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl-shared.zip"
        Write-Host "Downloading shared FFmpeg for TorchCodec..."
        Invoke-WebRequest -Uri $FfmpegUrl -OutFile $Archive
        Expand-Archive -LiteralPath $Archive -DestinationPath $SharedFfmpegRoot -Force
        Remove-Item -LiteralPath $Archive -Force
        $FfmpegExe = Get-ChildItem -LiteralPath $SharedFfmpegRoot -Filter "ffmpeg.exe" -File -Recurse |
            Select-Object -First 1
    }
    if (-not $FfmpegExe) {
        throw "Shared FFmpeg installation failed. Re-run without -SkipFfmpeg or install shared FFmpeg manually."
    }
    $env:Path = "$($FfmpegExe.DirectoryName);$env:Path"
}

$Check = "import torch, funasr, modelscope, streamlit; print('torch=' + torch.__version__); print('cuda=' + str(torch.cuda.is_available()))"
Invoke-Python @("-c", $Check)

if ($Pyannote) {
    $PyannoteCheck = "import sys; sys.path.insert(0, r'$($Root.Replace("'", "''"))\src'); from asr_mvp.diarization import ensure_windows_audio_dlls; ensure_windows_audio_dlls(); import pyannote.audio; print('pyannote=OK')"
    Invoke-Python @("-c", $PyannoteCheck)
}

Write-Host "Setup complete."
if ($UseCurrentEnvironment) {
    Write-Host "Start with: .\run_gui.ps1 -UseCurrentEnvironment"
} else {
    Write-Host "Start with: .\run_gui.ps1"
}
