[CmdletBinding()]
param([string]$RuntimeRoot)

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
$lockPath = Join-Path $stateRoot "controller.lock"
$controllerPid = $null
if (Test-Path -LiteralPath $lockPath) {
    $value = (Get-Content -Raw -LiteralPath $lockPath).Trim()
    if ($value -match '^\d+$') {
        $controllerPid = [int]$value
    }
}

if ($null -ne $controllerPid) {
    $process = Get-Process -Id $controllerPid -ErrorAction SilentlyContinue
    if ($null -ne $process) {
        Write-Host "Controller: running (PID=$controllerPid)"
    } else {
        Write-Host "Controller: not running (stale lock PID=$controllerPid)"
    }
} else {
    Write-Host "Controller: not running"
}

$statePath = Join-Path $stateRoot "state.json"
if (Test-Path -LiteralPath $statePath) {
    Get-Content -Raw -LiteralPath $statePath
} else {
    Write-Host "No continuous tuning Session has been created."
}

Write-Host "`nRecent controller log:"
$logPath = Join-Path $stateRoot "logs\controller\controller.log"
if (Test-Path -LiteralPath $logPath) {
    Get-Content -LiteralPath $logPath -Tail 20
}
