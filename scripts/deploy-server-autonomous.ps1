[CmdletBinding()]
param(
    [string]$RemoteHost = "hetao-npu",
    [string]$RemoteRoot = "/mnt/host-model/slai/user-1-wangakang/wangakang/cjx-workspace/vllmtkb-server-autonomous-418bd627-32c8cf190",
    [string]$AllowedWriteRoot = "/mnt/host-model/slai/user-1-wangakang/wangakang/cjx-workspace"
)

$ErrorActionPreference = "Stop"
$AllowedWriteRoot = $AllowedWriteRoot.TrimEnd("/")
if (-not $AllowedWriteRoot.StartsWith("/", [System.StringComparison]::Ordinal) -or $AllowedWriteRoot -eq "/") {
    throw "AllowedWriteRoot must be an absolute, non-root Linux directory."
}
$allowedPrefix = "$AllowedWriteRoot/"
if (-not $RemoteRoot.StartsWith($allowedPrefix, [System.StringComparison]::Ordinal)) {
    throw "RemoteRoot must remain inside AllowedWriteRoot ($AllowedWriteRoot)."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$archiveName = "vllmtkb-server-autonomous-$stamp.tar.gz"
$localArchive = Join-Path ([System.IO.Path]::GetTempPath()) $archiveName
$remoteArchive = "$allowedPrefix$archiveName"
$gitCommit = (& git -C $repoRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $gitCommit -notmatch '^[0-9a-f]{40}$') {
    throw "Failed to resolve the deployment Git commit."
}
$trackedChanges = & git -C $repoRoot status --porcelain --untracked-files=no
if ($LASTEXITCODE -ne 0) { throw "Failed to inspect the deployment worktree." }
if ($trackedChanges) {
    throw "Refusing to deploy tracked changes that are not committed."
}

# Archive exactly the committed tree. This prevents ignored private overlays,
# mutable runtime state, caches, and unrelated untracked files from leaking into
# the server snapshot.
& git -C $repoRoot archive --format=tar.gz --output=$localArchive $gitCommit
if ($LASTEXITCODE -ne 0) { throw "Failed to create the committed deployment archive." }
$localSize = (Get-Item -LiteralPath $localArchive).Length
$localSha256 = (Get-FileHash -LiteralPath $localArchive -Algorithm SHA256).Hash.ToLowerInvariant()

# Some managed SSH installations emit a host-key ownership warning and return
# nonzero after a command has completed. Validate the command's structured
# stdout instead of accepting or rejecting it from that transport quirk alone.
$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
& scp $localArchive "${RemoteHost}:$remoteArchive" 2>$null
$remoteIdentityOutput = & ssh -o BatchMode=yes $RemoteHost `
    "stat -c %s '$remoteArchive' && sha256sum '$remoteArchive'" 2>$null
$ErrorActionPreference = $previousErrorActionPreference
$remoteSize = ($remoteIdentityOutput | Where-Object { $_ -match '^\d+$' } | Select-Object -Last 1)
$remoteSha256 = ($remoteIdentityOutput | ForEach-Object {
    if ($_ -match '^([0-9a-fA-F]{64})\s+') { $Matches[1].ToLowerInvariant() }
} | Select-Object -Last 1)
if (-not $remoteSize -or [int64]$remoteSize -ne $localSize -or $remoteSha256 -ne $localSha256) {
    throw "Failed to verify uploaded deployment archive size and SHA-256."
}

$remoteCommand = @"
set -e
mkdir -p '$RemoteRoot'
tar -xzf '$remoteArchive' -C '$RemoteRoot'
chmod +x '$RemoteRoot'/tuning_pipeline/workflow/continuous/server_autonomous/*.sh
manifest='$RemoteRoot/deployment.identity.json'
temporary="`${manifest}.tmp-$stamp"
printf '%s\n' '{"schema":"vllmtkb-deployment/v1","git_commit":"$gitCommit","archive_sha256":"$localSha256","archive_size":$localSize,"deployed_at":"$((Get-Date).ToUniversalTime().ToString("o"))"}' > "`${temporary}"
mv "`${temporary}" "`${manifest}"
printf 'deployed_root=%s\nretained_archive=%s\ngit_commit=%s\narchive_sha256=%s\n' '$RemoteRoot' '$remoteArchive' '$gitCommit' '$localSha256'
printf '__AUTONOMOUS_DEPLOY_RC__=0\n'
"@
$ErrorActionPreference = "SilentlyContinue"
$remoteOutput = & ssh -o BatchMode=yes $RemoteHost $remoteCommand 2>&1
$ErrorActionPreference = $previousErrorActionPreference
$remoteOutput | Write-Host
if (-not ($remoteOutput -match '__AUTONOMOUS_DEPLOY_RC__=0')) {
    throw "Failed to extract autonomous deployment."
}

Write-Host "Local deployment archive retained at $localArchive"
