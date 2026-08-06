[CmdletBinding()]
param([string]$Config)

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
$arguments = @((Join-Path $root "continuous_tuning.py"), "--prepare-lab")
if ($Config) {
    if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) {
        throw "Config file does not exist: $Config"
    }
    $arguments += @("--config", (Resolve-Path -LiteralPath $Config).Path)
} else {
    $localConfig = Join-Path $root "config.local.yaml"
    if (Test-Path -LiteralPath $localConfig -PathType Leaf) {
        $arguments += @("--config", (Resolve-Path -LiteralPath $localConfig).Path)
    }
}
& $python @arguments
exit $LASTEXITCODE
