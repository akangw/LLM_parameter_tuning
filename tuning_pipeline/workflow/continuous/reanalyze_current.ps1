[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = if ($env:VLLMTKB_PYTHON) {
    if (-not (Test-Path -LiteralPath $env:VLLMTKB_PYTHON)) {
        throw "VLLMTKB_PYTHON does not exist: $env:VLLMTKB_PYTHON"
    }
    (Resolve-Path -LiteralPath $env:VLLMTKB_PYTHON).Path
} else {
    (Get-Command python -ErrorAction Stop).Source
}
& $python (Join-Path $root "continuous_tuning.py") --reanalyze-current
exit $LASTEXITCODE
