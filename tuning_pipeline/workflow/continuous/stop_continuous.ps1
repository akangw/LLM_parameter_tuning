[CmdletBinding()]
param([switch]$StopActiveTask)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$statePath = Join-Path $root "state.json"
$stopPath = Join-Path $root "STOP_REQUESTED"
"requested_at: $(Get-Date -Format o)" | Set-Content -LiteralPath $stopPath -Encoding utf8
Write-Host "Graceful stop requested. No next experiment will be submitted."

if ($StopActiveTask -and (Test-Path -LiteralPath $statePath)) {
    $state = Get-Content -Raw -LiteralPath $statePath | ConvertFrom-Json
    if ($state.active_task_id) {
        if ($state.execution_mode -eq "ktp_lab" -or $state.active_task_id -notmatch '^\d+$') {
            & ssh -o BatchMode=yes -o ConnectTimeout=15 hetao-npu `
                "cd /mnt/host-model/slai/user-1-wangakang/wangakang/cjx-workspace/vllmtkb-418bd627-32c8cf190 && ktp-lab stop --lease $($state.active_task_id)"
            Write-Host "Stop requested for persistent lease $($state.active_task_id)."
        } else {
            & ssh -o BatchMode=yes -o ConnectTimeout=15 hetao-npu "ktp stop $($state.active_task_id)"
            Write-Host "Stop requested for active task $($state.active_task_id)."
        }
    }
}
