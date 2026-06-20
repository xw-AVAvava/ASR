[CmdletBinding()]
param(
    [switch]$UseCurrentEnvironment
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvScripts = Join-Path $Root ".venv\Scripts"
$VenvPython = Join-Path $VenvScripts "python.exe"

if (-not $UseCurrentEnvironment -and (Test-Path $VenvPython)) {
    $Python = $VenvPython
    $env:Path = "$VenvScripts;$env:Path"
} else {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonCommand) {
        throw "Python was not found. Run setup.ps1 or activate your Python environment first."
    }
    $Python = $PythonCommand.Source
}

$FfmpegRoots = @(
    (Join-Path $Root ".runtime\ffmpeg-shared"),
    (Join-Path $Root ".venv\ffmpeg-shared")
)
foreach ($FfmpegRoot in $FfmpegRoots) {
    if (-not (Test-Path $FfmpegRoot)) {
        continue
    }
    $FfmpegExe = Get-ChildItem -LiteralPath $FfmpegRoot -Filter "ffmpeg.exe" -File -Recurse -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($FfmpegExe) {
        $env:Path = "$($FfmpegExe.DirectoryName);$env:Path"
        break
    }
}

$env:PYTHONUTF8 = "1"
Set-Location $Root
Write-Host "Using Python: $Python"
& $Python -m streamlit run (Join-Path $Root "app.py")
