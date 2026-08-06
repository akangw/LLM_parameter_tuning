[CmdletBinding()]
param(
    [switch]$Foreground,
    [switch]$Resume,
    [switch]$RetryPausedCurrent,
    [switch]$NewSession,
    [switch]$CheckOnly,
    [string]$StrategyProfile,
    [string]$BenchmarkProfile,
    [string]$SearchSpaceProfile,
    [ValidateSet("codex", "anthropic", "openai_compatible", "deepseek", "command")]
    [string]$AgentProvider,
    [string]$Config
)

$ErrorActionPreference = "Stop"
$entry = Join-Path $PSScriptRoot "..\tuning_pipeline\workflow\continuous\start_continuous.ps1"
& $entry @PSBoundParameters
exit $LASTEXITCODE
