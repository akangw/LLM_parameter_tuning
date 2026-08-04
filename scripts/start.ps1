[CmdletBinding()]
param(
    [switch]$Foreground,
    [switch]$Resume,
    [switch]$RetryPausedCurrent,
    [switch]$NewSession,
    [switch]$CheckOnly,
    [string]$StrategyProfile,
    [ValidateSet("codex", "anthropic", "openai_compatible", "command")]
    [string]$AgentProvider
)

$ErrorActionPreference = "Stop"
$entry = Join-Path $PSScriptRoot "..\tuning_pipeline\workflow\continuous\start_continuous.ps1"
& $entry @PSBoundParameters
exit $LASTEXITCODE
