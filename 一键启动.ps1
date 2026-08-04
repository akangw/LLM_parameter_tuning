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
& (Join-Path $PSScriptRoot "scripts\start.ps1") @PSBoundParameters
exit $LASTEXITCODE
