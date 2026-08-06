param(
    [Parameter(Position=0)][ValidateSet("validate", "approve")][string]$Command = "validate",
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments
)
$root = Split-Path -Parent $PSScriptRoot
& python "$root/tuning_pipeline/workflow/continuous/image_identity_cli.py" $Command @Arguments
exit $LASTEXITCODE
