[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Vllm,
    [Parameter(Mandatory=$true)][string]$VllmAscend,
    [ValidateSet("codex", "anthropic")][string]$Provider = "codex",
    [switch]$PrepareOnly,
    [switch]$Resume,
    [int]$Concurrency = 8
)

$ErrorActionPreference = "Stop"
$arguments = @(
    (Join-Path $PSScriptRoot "migrate_versions.py"),
    "--vllm", $Vllm,
    "--vllm-ascend", $VllmAscend,
    "--provider", $Provider,
    "--concurrency", $Concurrency
)
if ($PrepareOnly) { $arguments += "--prepare-only" }
if ($Resume) { $arguments += "--resume" }
& python @arguments
exit $LASTEXITCODE
