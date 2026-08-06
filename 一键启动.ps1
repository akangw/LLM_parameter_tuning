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
    [string]$Config,
    [string]$RuntimeRoot
)

$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "scripts\start.ps1") @PSBoundParameters
exit $LASTEXITCODE
