$ErrorActionPreference = "Stop"
$entry = Join-Path $PSScriptRoot "..\tuning_pipeline\workflow\continuous\prepare_lab.ps1"
& $entry
exit $LASTEXITCODE
