[CmdletBinding()]
param(
    [switch]$Foreground,
    [switch]$Resume,
    [switch]$RetryPausedCurrent,
    [switch]$NewSession,
    [switch]$CheckOnly,
    [string]$StrategyProfile,
    [string]$BenchmarkProfile,
    [ValidateSet("codex", "anthropic", "openai_compatible", "command")]
    [string]$AgentProvider
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$downstreamRoot = Split-Path -Parent (Split-Path -Parent $root)
$pipelineRoot = Split-Path -Parent $downstreamRoot

if ((@($Resume, $RetryPausedCurrent, $NewSession) | Where-Object { $_ }).Count -gt 1) {
    throw "-Resume, -RetryPausedCurrent and -NewSession are mutually exclusive."
}
if (($Resume -or $RetryPausedCurrent) -and ($StrategyProfile -or $AgentProvider -or $BenchmarkProfile)) {
    throw "Strategy/Agent/Benchmark profiles are frozen in a Session and cannot be overridden during resume."
}

function Get-ControllerProcess {
    $candidatePids = @()
    foreach ($pidName in @("controller.lock", "controller.pid")) {
        $pidPath = Join-Path $root $pidName
        if (Test-Path -LiteralPath $pidPath) {
            $value = (Get-Content -Raw -LiteralPath $pidPath).Trim()
            if ($value -match '^\d+$') {
                $candidatePids += [int]$value
            }
        }
    }
    foreach ($candidatePid in ($candidatePids | Select-Object -Unique)) {
        $process = Get-Process -Id $candidatePid -ErrorAction SilentlyContinue
        if ($null -eq $process) {
            continue
        }
        $commandLine = (Get-CimInstance Win32_Process `
            -Filter "ProcessId = $candidatePid" -ErrorAction SilentlyContinue).CommandLine
        if ($commandLine -and $commandLine -match 'continuous_tuning\.py') {
            return $process
        }
    }
    return $null
}

function Resolve-Python {
    $override = $env:VLLMTKB_PYTHON
    if ($override) {
        if (-not (Test-Path -LiteralPath $override)) {
            throw "VLLMTKB_PYTHON does not exist: $override"
        }
        return (Resolve-Path -LiteralPath $override).Path
    }
    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        throw "Python was not found. Install Python 3.11+ or set VLLMTKB_PYTHON."
    }
    return $command.Source
}

$running = Get-ControllerProcess
if ($null -ne $running) {
    Write-Host "Controller is already running. PID=$($running.Id)"
    Write-Host "No second controller was started."
    exit 0
}

$requiredFiles = @(
    "continuous_tuning.py",
    "config.yaml",
    "agent_provider.py",
    "strategy_profiles.yaml",
    "benchmark_profiles.yaml",
    "agent_decision.schema.json",
    "failure_decision.schema.json",
    "remote\image_version_manifest.yaml"
)
foreach ($relativePath in $requiredFiles) {
    $fullPath = Join-Path $root $relativePath
    if (-not (Test-Path -LiteralPath $fullPath)) {
        throw "Required file is missing: $fullPath"
    }
}

$portraitIndexPath = Join-Path $pipelineRoot "portrait_pipeline\build\codex_portrait_pipeline\run\index.json"
$portraitOutputPath = Join-Path $pipelineRoot "portrait_pipeline\outputs\ParameterYAML"
if (Test-Path -LiteralPath $portraitIndexPath) {
    $portraitIndex = Get-Content -Raw -LiteralPath $portraitIndexPath | ConvertFrom-Json
    $portraitUnfinished = [int]$portraitIndex.summary.pending +
        [int]$portraitIndex.summary.in_progress +
        [int]$portraitIndex.summary.error
    if ($portraitUnfinished -gt 0) {
        throw "Portrait migration is not complete: $portraitUnfinished unfinished tasks."
    }
} elseif (
    -not (Test-Path -LiteralPath $portraitOutputPath) -or
    (Get-ChildItem -LiteralPath $portraitOutputPath -Filter *.yaml -File).Count -eq 0
) {
    throw "Neither a completed portrait queue nor formal ParameterYAML artifacts are available."
}

$tagProgressPath = Join-Path $downstreamRoot "tag_params\output\progress.json"
if (-not (Test-Path -LiteralPath $tagProgressPath)) {
    throw "Codex tags have not been generated: $tagProgressPath"
}
$tagProgress = Get-Content -Raw -LiteralPath $tagProgressPath | ConvertFrom-Json
$tagUnfinished = [int]$tagProgress.summary.pending +
    [int]$tagProgress.summary.in_progress +
    [int]$tagProgress.summary.error
if ($tagUnfinished -gt 0) {
    throw "Codex tags are not complete: $tagUnfinished unfinished items."
}

$activationPath = Join-Path $root "activation.approved.yaml"
if (-not (Test-Path -LiteralPath $activationPath)) {
    throw @"
Remote activation is intentionally locked. Verify the target image actually
contains vLLM 418bd6273c03bf48d5066733769e0a74bdc51694 and vllm-ascend
32c8cf190f596b47f0d0b965e64aea9f2b789ad4, then create:
$activationPath
from activation.example.yaml.
"@
}
$activationText = Get-Content -Raw -LiteralPath $activationPath
if ($activationText -notmatch '(?m)^approved:\s*true\s*$') {
    throw "Remote activation file does not contain 'approved: true': $activationPath"
}
foreach ($requiredIdentity in @(
    "418bd6273c03bf48d5066733769e0a74bdc51694",
    "32c8cf190f596b47f0d0b965e64aea9f2b789ad4"
)) {
    if ($activationText -notmatch [regex]::Escape($requiredIdentity)) {
        throw "Remote activation evidence is missing commit $requiredIdentity"
    }
}

$python = Resolve-Python
& $python -c "from importlib.metadata import version; from packaging.version import Version; import yaml, paramiko, pydantic, jsonschema, anthropic; required={'PyYAML':'6.0','paramiko':'3.0','pydantic':'2.0','jsonschema':'4.0','anthropic':'0.49','packaging':'23.0'}; bad=[f'{name}={version(name)} (need >= {minimum})' for name,minimum in required.items() if Version(version(name)) < Version(minimum)]; assert not bad, '; '.join(bad); print('Python dependencies: OK')"
if ($LASTEXITCODE -ne 0) {
    throw "Python dependencies are incomplete. Run from the project root: python -m pip install -r .\tuning_pipeline\requirements-runtime.txt"
}

$configText = Get-Content -Raw -LiteralPath (Join-Path $root "config.yaml")
$effectiveProvider = if ($AgentProvider) { $AgentProvider } elseif ($configText -match '(?ms)^agent:\s*.*?^\s+provider:\s*([^\s#]+)') { $Matches[1] } else { "codex" }
if ($effectiveProvider -eq "codex") {
    $codex = Get-Command codex.cmd -ErrorAction SilentlyContinue
    if ($null -eq $codex) {
        $codex = Get-Command codex -ErrorAction SilentlyContinue
    }
    if (($null -eq $codex) -and (-not $env:VLLMTKB_CODEX_COMMAND)) {
        throw "Codex CLI was not found. Put it on PATH or set VLLMTKB_CODEX_COMMAND."
    }
}

$mode = "--start"
if ($RetryPausedCurrent) {
    $mode = "--retry-paused-current"
} elseif ($Resume) {
    $mode = "--resume"
} elseif (-not $NewSession) {
    $statePath = Join-Path $root "state.json"
    if (Test-Path -LiteralPath $statePath) {
        $state = Get-Content -Raw -LiteralPath $statePath | ConvertFrom-Json
        $resumableStatuses = @(
            "initialized",
            "running",
            "stop_requested",
            "paused_controller_error",
            "stopped_after_current_round",
            "stopped_after_failed_round"
        )
        if ($state.status -eq "paused_for_human") {
            throw "The last Session is paused for human review. Inspect state.json before resuming."
        }
        if ($state.active_run_id -and ($resumableStatuses -contains $state.status)) {
            $mode = "--resume"
        }
    }
}

Write-Host "Preflight: OK"
Write-Host "Recommended launch mode: $mode"
if ($CheckOnly) {
    $checkArguments = @((Join-Path $root "continuous_tuning.py"), "--check-only")
    if ($StrategyProfile) { $checkArguments += @("--strategy-profile", $StrategyProfile) }
    if ($AgentProvider) { $checkArguments += @("--agent-provider", $AgentProvider) }
    if ($BenchmarkProfile) { $checkArguments += @("--benchmark-profile", $BenchmarkProfile) }
    & $python @checkArguments
    if ($LASTEXITCODE -ne 0) {
        $checkExitCode = $LASTEXITCODE
        Write-Host "End-to-end check failed; no controller or experiment was started." -ForegroundColor Red
        exit $checkExitCode
    }
    Write-Host "Check-only mode: local config, AI provider, SSH, and idle Lease are ready."
    Write-Host "No controller was started and no local or remote files were changed."
    exit 0
}

$stopFile = Join-Path $root "STOP_REQUESTED"
if (Test-Path -LiteralPath $stopFile) {
    $archived = Join-Path $root ("STOP_REQUESTED." + (Get-Date -Format "yyyyMMdd_HHmmss"))
    Move-Item -LiteralPath $stopFile -Destination $archived
}

$arguments = @((Join-Path $root "continuous_tuning.py"), $mode)
if ($StrategyProfile) { $arguments += @("--strategy-profile", $StrategyProfile) }
if ($AgentProvider) { $arguments += @("--agent-provider", $AgentProvider) }
if ($BenchmarkProfile) { $arguments += @("--benchmark-profile", $BenchmarkProfile) }
if ($Foreground) {
    & $python @arguments
    exit $LASTEXITCODE
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$processLogDir = Join-Path $root "logs\controller\process"
New-Item -ItemType Directory -Path $processLogDir -Force | Out-Null
$stdout = Join-Path $processLogDir "controller_process_$timestamp.stdout.log"
$stderr = Join-Path $processLogDir "controller_process_$timestamp.stderr.log"
$process = Start-Process -FilePath $python -ArgumentList $arguments `
    -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$process.Id | Set-Content -LiteralPath (Join-Path $root "controller.pid") -Encoding ascii
@(
    "pid=$($process.Id)"
    "mode=$mode"
    "started_at=$(Get-Date -Format o)"
    "stdout=$stdout"
    "stderr=$stderr"
) | Set-Content -LiteralPath (Join-Path $root "latest_controller_launch.txt") -Encoding utf8

Write-Host "Continuous controller started. PID=$($process.Id), mode=$mode"
Write-Host "State: $(Join-Path $root 'state.json')"
