param(
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments
)
$root = Split-Path -Parent $PSScriptRoot
& python "$root/tuning_pipeline/workflow/continuous/runtime_adapter_cli.py" @Arguments
exit $LASTEXITCODE
