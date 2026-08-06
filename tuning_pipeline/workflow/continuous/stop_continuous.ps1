[CmdletBinding()]
param([switch]$StopActiveTask, [string]$RuntimeRoot)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$stateRoot = if ($RuntimeRoot) {
    if ([System.IO.Path]::IsPathRooted($RuntimeRoot)) {
        [System.IO.Path]::GetFullPath($RuntimeRoot)
    } else {
        [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $RuntimeRoot))
    }
} else {
    $root
}
New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null
$statePath = Join-Path $stateRoot "state.json"
$stopPath = Join-Path $stateRoot "STOP_REQUESTED"
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
    $arguments = @((Join-Path $root "continuous_tuning.py"), "--stop-active-task")
    if ($RuntimeRoot) { $arguments += @("--runtime-root", $stateRoot) }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
