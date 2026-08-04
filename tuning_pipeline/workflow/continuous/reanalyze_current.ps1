[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
& python (Join-Path $root "continuous_tuning.py") --reanalyze-current
exit $LASTEXITCODE
