param(
    [int]$IntervalSeconds = 5,
    [switch]$Once
)

$root = $PSScriptRoot
$statusPath = Join-Path $root "build-status.json"
$progressPath = Join-Path $root "tag_params\output\progress.json"

do {
    Clear-Host
    Write-Host "Parameter knowledge build" -ForegroundColor Cyan
    if (Test-Path -LiteralPath $statusPath) {
        $status = Get-Content -Raw -LiteralPath $statusPath | ConvertFrom-Json
        Write-Host ("Stage:      {0}" -f $status.state)
        Write-Host ("Updated:    {0}" -f $status.updated_at)
        if ($status.error) {
            Write-Host ("Error:      {0}" -f $status.error) -ForegroundColor Red
        }
    } else {
        Write-Host "Stage:      not started"
    }

    if (Test-Path -LiteralPath $progressPath) {
        $progress = Get-Content -Raw -LiteralPath $progressPath | ConvertFrom-Json
        $summary = $progress.summary
        $percent = if ([int]$summary.total -gt 0) {
            [math]::Round(100 * [int]$summary.completed / [int]$summary.total, 2)
        } else { 0 }
        Write-Host ""
        Write-Host "Codex tags" -ForegroundColor Cyan
        Write-Host ("Progress:   {0}/{1} ({2}%)" -f $summary.completed, $summary.total, $percent)
        Write-Host ("Pending:    {0}" -f $summary.pending)
        Write-Host ("Running:    {0}" -f $summary.in_progress)
        Write-Host ("Errors:     {0}" -f $summary.error)
    }
    if (-not $Once) {
        Write-Host ""
        Write-Host ("Refreshing every {0}s. Press Ctrl+C to stop watching." -f $IntervalSeconds) `
            -ForegroundColor DarkGray
        Start-Sleep -Seconds $IntervalSeconds
    }
} while (-not $Once)
