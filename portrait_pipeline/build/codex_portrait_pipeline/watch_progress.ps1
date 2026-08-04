param(
    [int]$IntervalSeconds = 5,
    [switch]$Once
)

$pipelineRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$runDir = Join-Path $pipelineRoot "run"
$indexPath = Join-Path $runDir "index.json"
$supervisorPath = Join-Path $runDir "supervisor-status.json"

function Show-PortraitProgress {
    if (-not (Test-Path -LiteralPath $indexPath)) {
        Write-Host "Queue index not found: $indexPath" -ForegroundColor Red
        return
    }

    $index = Get-Content -LiteralPath $indexPath -Raw | ConvertFrom-Json
    $summary = $index.summary
    $processed = [int]$summary.completed + [int]$summary.skipped
    $percent = if ([int]$summary.total -gt 0) {
        [math]::Round(100 * $processed / [int]$summary.total, 2)
    } else {
        0
    }

    $supervisor = $null
    if (Test-Path -LiteralPath $supervisorPath) {
        $supervisor = Get-Content -LiteralPath $supervisorPath -Raw | ConvertFrom-Json
    }

    $latest = @(
        $index.tasks |
            Where-Object { $_.status -in @("completed", "skipped") } |
            Sort-Object sequence -Descending |
            Select-Object -First 5
    )

    Clear-Host
    Write-Host "Codex ParameterYAML progress" -ForegroundColor Cyan
    Write-Host ("Updated:       {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    Write-Host ("Progress:      {0}/{1} ({2}%)" -f $processed, $summary.total, $percent)
    Write-Host ("Completed:     {0}" -f $summary.completed)
    Write-Host ("Skipped:       {0}" -f $summary.skipped)
    Write-Host ("Pending:       {0}" -f $summary.pending)
    Write-Host ("In progress:   {0}" -f $summary.in_progress)
    Write-Host ("Errors:        {0}" -f $summary.error)

    if ($supervisor) {
        $processAlive = $false
        if ($supervisor.PSObject.Properties.Name -contains "pid") {
            $processAlive = $null -ne (Get-Process -Id $supervisor.pid -ErrorAction SilentlyContinue)
        } else {
            $processAlive = $null -ne (
                Get-CimInstance Win32_Process |
                    Where-Object {
                        $_.CommandLine -like "*codex_portrait_pipeline*supervisor.py*"
                    } |
                    Select-Object -First 1
            )
        }

        Write-Host ""
        Write-Host "Current worker" -ForegroundColor Cyan
        Write-Host ("Supervisor:    {0} (process alive: {1})" -f $supervisor.state, $processAlive)
        Write-Host ("Sequence:      {0}" -f $supervisor.sequence)
        Write-Host ("Parameter:     {0}" -f $supervisor.name)
        Write-Host ("Task ID:       {0}" -f $supervisor.task_id)
        Write-Host ("Started UTC:   {0}" -f $supervisor.started_at)
    }

    Write-Host ""
    Write-Host "Latest results" -ForegroundColor Cyan
    foreach ($task in $latest) {
        Write-Host ("#{0,-4} {1,-10} {2}" -f $task.sequence, $task.status, $task.name)
    }

    if (-not $Once) {
        Write-Host ""
        Write-Host ("Refreshing every {0}s. Press Ctrl+C to stop." -f $IntervalSeconds) -ForegroundColor DarkGray
    }
}

do {
    Show-PortraitProgress
    if (-not $Once) {
        Start-Sleep -Seconds $IntervalSeconds
    }
} while (-not $Once)
