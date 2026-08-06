[CmdletBinding()]
param([switch]$StopActiveTask, [string]$RuntimeRoot)

$ErrorActionPreference = "Stop"
$entry = Join-Path $PSScriptRoot "..\tuning_pipeline\workflow\continuous\stop_continuous.ps1"
& $entry @PSBoundParameters
exit $LASTEXITCODE
