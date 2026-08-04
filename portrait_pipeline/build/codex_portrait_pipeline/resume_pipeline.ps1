$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$supervisor = Join-Path $projectRoot "codex_portrait_pipeline\supervisor.py"

$existing = Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -eq "python.exe" -and
        $_.CommandLine -like "*codex_portrait_pipeline*supervisor.py*"
    } |
    Select-Object -First 1

if ($existing) {
    Write-Host ("Portrait pipeline is already running (PID {0})." -f $existing.ProcessId)
    exit 0
}

$python = Get-Command python -ErrorAction Stop
$process = Start-Process `
    -FilePath $python.Source `
    -ArgumentList @($supervisor) `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -PassThru

Start-Sleep -Seconds 2
if (Get-Process -Id $process.Id -ErrorAction SilentlyContinue) {
    Write-Host ("Portrait pipeline resumed successfully (PID {0})." -f $process.Id)
    Write-Host "Interrupted in-progress tasks are automatically returned to the queue."
    exit 0
}

Write-Error "Portrait pipeline failed to start. Check build\codex_portrait_pipeline\run\supervisor-status.json."
exit 1
