[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Vllm,
    [Parameter(Mandatory=$true)][string]$VllmAscend,
    [ValidateSet("codex", "anthropic")][string]$Provider = "codex",
    [ValidateSet("migrate", "rebuild")][string]$PortraitMode = "migrate",
    [string]$LegacyPortraitDir,
    [ValidateSet("codex", "anthropic", "openai_compatible", "command")]
    [string]$TagProvider,
    [string]$Scenario,
    [switch]$PrepareOnly,
    [switch]$Resume,
    [int]$Concurrency = 8
)

$ErrorActionPreference = "Stop"
$arguments = @(
    (Join-Path $PSScriptRoot "migrate_versions.py"),
    "--vllm", $Vllm,
    "--vllm-ascend", $VllmAscend,
    "--provider", $Provider,
    "--portrait-mode", $PortraitMode,
    "--concurrency", $Concurrency
)
if ($PrepareOnly) { $arguments += "--prepare-only" }
if ($Resume) { $arguments += "--resume" }
if ($TagProvider) { $arguments += @("--tag-provider", $TagProvider) }
if ($Scenario) { $arguments += @("--scenario", $Scenario) }
if ($LegacyPortraitDir) { $arguments += @("--legacy-dir", $LegacyPortraitDir) }
$python = if ($env:VLLMTKB_PYTHON) {
    if (-not (Test-Path -LiteralPath $env:VLLMTKB_PYTHON)) {
        throw "VLLMTKB_PYTHON does not exist: $env:VLLMTKB_PYTHON"
    }
    (Resolve-Path -LiteralPath $env:VLLMTKB_PYTHON).Path
} else {
    (Get-Command python -ErrorAction Stop).Source
}
& $python @arguments
exit $LASTEXITCODE
