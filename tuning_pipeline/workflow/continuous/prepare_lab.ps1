[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
& python (Join-Path $root "continuous_tuning.py") --prepare-lab
exit $LASTEXITCODE
