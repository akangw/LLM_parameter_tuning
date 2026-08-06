[CmdletBinding()]
param([string]$RuntimeRoot)

$ErrorActionPreference = "Stop"
$entry = Join-Path $PSScriptRoot "..\tuning_pipeline\workflow\continuous\status_continuous.ps1"
& $entry @PSBoundParameters
exit $LASTEXITCODE
