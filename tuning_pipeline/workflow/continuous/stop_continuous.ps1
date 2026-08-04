[CmdletBinding()]
param([switch]$StopActiveTask)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$statePath = Join-Path $root "state.json"
$stopPath = Join-Path $root "STOP_REQUESTED"
"requested_at: $(Get-Date -Format o)" | Set-Content -LiteralPath $stopPath -Encoding utf8
Write-Host "Graceful stop requested. No next experiment will be submitted."

if ($StopActiveTask -and (Test-Path -LiteralPath $statePath)) {
    $python = if ($env:VLLMTKB_PYTHON) {
        if (-not (Test-Path -LiteralPath $env:VLLMTKB_PYTHON)) {
            throw "VLLMTKB_PYTHON does not exist: $env:VLLMTKB_PYTHON"
        }
        (Resolve-Path -LiteralPath $env:VLLMTKB_PYTHON).Path
    } else {
        (Get-Command python -ErrorAction Stop).Source
    }
    & $python (Join-Path $root "continuous_tuning.py") --stop-active-task
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
