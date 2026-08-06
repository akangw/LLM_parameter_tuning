[CmdletBinding()]
param(
    [ValidateSet("list", "show", "validate", "init", "check", "prepare", "start", "resume", "status", "stop")]
    [string]$Action = "list",
    [string]$Name,
    [string]$Config,
    [string]$RuntimeRoot,
    [string]$StrategyProfile,
    [string]$BenchmarkProfile,
    [string]$SearchSpaceProfile,
    [ValidateSet("codex", "anthropic", "openai_compatible", "deepseek", "command")]
    [string]$AgentProvider,
    [switch]$Foreground,
    [switch]$StopActiveTask
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$manager = Join-Path $repoRoot "scenarios\manage.py"
$python = (Get-Command python -ErrorAction Stop).Source

if ($Action -eq "list") {
    & $python $manager list
    exit $LASTEXITCODE
}
if (-not $Name) {
    throw "-Name is required for action '$Action'. Run -Action list first."
}
if ($Action -eq "show") {
    & $python $manager show $Name
    exit $LASTEXITCODE
}
if ($Action -eq "validate") {
    & $python $manager validate $Name
    exit $LASTEXITCODE
}

$json = & $python $manager resolve $Name
if ($LASTEXITCODE -ne 0) {
    throw "Scenario validation failed: $json"
}
$scenario = $json | ConvertFrom-Json

if ($Action -eq "init") {
    $target = $scenario.operator_local_config
    if (Test-Path -LiteralPath $target) {
        throw "Operator config already exists; refusing to overwrite it: $target"
    }
    Copy-Item -LiteralPath $scenario.operator_template -Destination $target
    Write-Host "Created ignored operator config: $target"
    Write-Host "Replace every CHANGE_ME value, then run: .\scripts\scenario.ps1 -Action check -Name $Name"
    exit 0
}

$selectedConfig = if ($Config) { (Resolve-Path -LiteralPath $Config).Path } else { $scenario.selected_config }
$selectedRuntime = if ($RuntimeRoot) { $RuntimeRoot } else { $scenario.runtime_root }
$profileArguments = @{}
foreach ($item in @("StrategyProfile", "BenchmarkProfile", "SearchSpaceProfile", "AgentProvider")) {
    $value = Get-Variable -Name $item -ValueOnly
    if ($value) { $profileArguments[$item] = $value }
}

if ($Action -in @("prepare", "start", "resume") -and $scenario.status -ne "integrated") {
    throw "Scenario '$Name' is $($scenario.status), not integrated. Complete its readiness attestations first."
}

switch ($Action) {
    "check" {
        & (Join-Path $PSScriptRoot "start.ps1") -Config $selectedConfig -RuntimeRoot $selectedRuntime -CheckOnly -NewSession @profileArguments
    }
    "prepare" {
        & (Join-Path $PSScriptRoot "prepare-remote.ps1") -Config $selectedConfig @profileArguments
    }
    "start" {
        & (Join-Path $PSScriptRoot "start.ps1") -Config $selectedConfig -RuntimeRoot $selectedRuntime -NewSession -Foreground:$Foreground @profileArguments
    }
    "resume" {
        & (Join-Path $PSScriptRoot "start.ps1") -Config $selectedConfig -RuntimeRoot $selectedRuntime -Resume -Foreground:$Foreground @profileArguments
    }
    "status" {
        & (Join-Path $PSScriptRoot "status.ps1") -RuntimeRoot $selectedRuntime
    }
    "stop" {
        & (Join-Path $PSScriptRoot "stop.ps1") -RuntimeRoot $selectedRuntime -StopActiveTask:$StopActiveTask
    }
}
exit $LASTEXITCODE
