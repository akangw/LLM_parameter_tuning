[CmdletBinding()]
param(
    [string]$Config,
    [string]$StrategyProfile,
    [string]$BenchmarkProfile,
    [string]$SearchSpaceProfile,
    [ValidateSet("codex", "anthropic", "openai_compatible", "deepseek", "command")]
    [string]$AgentProvider
)

$ErrorActionPreference = "Stop"
$entry = Join-Path $PSScriptRoot "..\tuning_pipeline\workflow\continuous\prepare_lab.ps1"
& $entry @PSBoundParameters
exit $LASTEXITCODE
