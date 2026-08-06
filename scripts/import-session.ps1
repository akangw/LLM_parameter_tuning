param(
    [Parameter(Mandatory=$true)][string]$Bundle,
    [Parameter(Mandatory=$true)][string]$RuntimeRoot,
    [switch]$Activate
)
$root = Split-Path -Parent $PSScriptRoot
$argsList = @("$root/tuning_pipeline/workflow/continuous/session_bundle.py", "import", $Bundle, "--runtime-root", $RuntimeRoot)
if ($Activate) { $argsList += "--activate" }
& python @argsList
exit $LASTEXITCODE
