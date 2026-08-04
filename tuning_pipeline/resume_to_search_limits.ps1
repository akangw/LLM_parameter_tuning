$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$runner = Join-Path $root "knowledge_pipeline.py"
$pidPath = Join-Path $root "knowledge-pipeline.pid"

$existing = $null
if (Test-Path -LiteralPath $pidPath) {
    $pidValue = (Get-Content -Raw -LiteralPath $pidPath).Trim()
    if ($pidValue -match '^\d+$') {
        $candidate = Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" `
            -ErrorAction SilentlyContinue
        if ($candidate -and $candidate.CommandLine -like "*knowledge_pipeline.py*") {
            $existing = $candidate
        }
    }
}
if ($existing) {
    Write-Host ("Knowledge pipeline is already running (PID {0})." -f $existing.ProcessId)
    exit 0
}

$python = Get-Command python -ErrorAction Stop
$process = Start-Process `
    -FilePath $python.Source `
    -ArgumentList @($runner) `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $root "knowledge-pipeline.stdout.log") `
    -RedirectStandardError (Join-Path $root "knowledge-pipeline.stderr.log") `
    -PassThru
$process.Id | Set-Content -LiteralPath $pidPath -Encoding ascii

Start-Sleep -Seconds 2
if (Get-Process -Id $process.Id -ErrorAction SilentlyContinue) {
    Write-Host ("Knowledge pipeline started (PID {0})." -f $process.Id)
    Write-Host "It will stop after Search Limits are compiled."
    exit 0
}

Write-Error "Knowledge pipeline failed to stay running. Check knowledge-pipeline.stderr.log."
