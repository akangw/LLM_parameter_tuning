[CmdletBinding()]
param(
    [switch]$Foreground,
    [switch]$Resume,
    [switch]$RetryPausedCurrent,
    [switch]$NewSession,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "scripts\start.ps1") @PSBoundParameters
exit $LASTEXITCODE
