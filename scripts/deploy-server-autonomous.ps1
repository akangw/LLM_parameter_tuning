[CmdletBinding()]
param(
    [string]$RemoteHost = "hetao-npu",
    [string]$RemoteRoot = "/mnt/host-model/slai/user-1-wangakang/wangakang/cjx-workspace/vllmtkb-server-autonomous-418bd627-32c8cf190"
)

$ErrorActionPreference = "Stop"
$allowedPrefix = "/mnt/host-model/slai/user-1-wangakang/wangakang/cjx-workspace/"
if (-not $RemoteRoot.StartsWith($allowedPrefix, [System.StringComparison]::Ordinal)) {
    throw "RemoteRoot must remain inside the approved cjx-workspace directory."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$archiveName = "vllmtkb-server-autonomous-$stamp.tar.gz"
$localArchive = Join-Path ([System.IO.Path]::GetTempPath()) $archiveName
$remoteArchive = "$allowedPrefix$archiveName"

& tar -czf $localArchive `
    --exclude=.git `
    --exclude=.pytest_cache `
    --exclude='*/__pycache__' `
    --exclude='*.pyc' `
    --exclude='portrait_pipeline/sources' `
    --exclude='tuning_pipeline/workflow/continuous/experiments' `
    --exclude='tuning_pipeline/workflow/continuous/logs' `
    --exclude='tuning_pipeline/workflow/continuous/state.json' `
    --exclude='tuning_pipeline/workflow/continuous/server_autonomous/runtime' `
    -C $repoRoot .
if ($LASTEXITCODE -ne 0) { throw "Failed to create deployment archive." }

& scp $localArchive "${RemoteHost}:$remoteArchive"
$localSize = (Get-Item -LiteralPath $localArchive).Length
$remoteSizeOutput = & ssh -o BatchMode=yes $RemoteHost "stat -c %s '$remoteArchive' 2>/dev/null || true" 2>$null
$remoteSize = ($remoteSizeOutput | Where-Object { $_ -match '^\d+$' } | Select-Object -Last 1)
if (-not $remoteSize -or [int64]$remoteSize -ne $localSize) {
    throw "Failed to verify uploaded deployment archive size."
}

$remoteCommand = @"
set -e
mkdir -p '$RemoteRoot'
tar -xzf '$remoteArchive' -C '$RemoteRoot'
chmod +x '$RemoteRoot'/tuning_pipeline/workflow/continuous/server_autonomous/*.sh
printf 'deployed_root=%s\nretained_archive=%s\n' '$RemoteRoot' '$remoteArchive'
printf '__AUTONOMOUS_DEPLOY_RC__=0\n'
"@
$remoteOutput = & ssh -o BatchMode=yes $RemoteHost $remoteCommand 2>&1
$remoteOutput | Write-Host
if (-not ($remoteOutput -match '__AUTONOMOUS_DEPLOY_RC__=0')) {
    throw "Failed to extract autonomous deployment."
}

Write-Host "Local deployment archive retained at $localArchive"
