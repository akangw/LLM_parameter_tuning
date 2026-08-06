param(
    [Parameter(Mandatory=$true)][string]$RuntimeRoot,
    [Parameter(Mandatory=$true)][string]$Output,
    [string]$SessionDir,
    [switch]$AllowActiveSnapshot
)
$root = Split-Path -Parent $PSScriptRoot
$argsList = @("$root/tuning_pipeline/workflow/continuous/session_bundle.py", "export", "--runtime-root", $RuntimeRoot, "--output", $Output)
if ($SessionDir) { $argsList += @("--session-dir", $SessionDir) }
if ($AllowActiveSnapshot) { $argsList += "--allow-active-snapshot" }
& python @argsList
exit $LASTEXITCODE
