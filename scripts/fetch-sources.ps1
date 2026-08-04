[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$sourceRoot = Join-Path $PSScriptRoot "..\portrait_pipeline\sources"
$repositories = @(
    @{
        Name = "vllm"
        Url = "https://github.com/vllm-project/vllm.git"
        Commit = "418bd6273c03bf48d5066733769e0a74bdc51694"
    },
    @{
        Name = "vllm-ascend"
        Url = "https://github.com/vllm-project/vllm-ascend.git"
        Commit = "32c8cf190f596b47f0d0b965e64aea9f2b789ad4"
    }
)

New-Item -ItemType Directory -Path $sourceRoot -Force | Out-Null
foreach ($repository in $repositories) {
    $target = Join-Path $sourceRoot $repository.Name
    if (-not (Test-Path -LiteralPath $target)) {
        & git clone --filter=blob:none --no-checkout $repository.Url $target
        if ($LASTEXITCODE -ne 0) { throw "Clone failed: $($repository.Url)" }
    }
    & git -C $target fetch --filter=blob:none origin $repository.Commit
    if ($LASTEXITCODE -ne 0) { throw "Fetch failed: $($repository.Commit)" }
    & git -C $target checkout --detach $repository.Commit
    if ($LASTEXITCODE -ne 0) { throw "Checkout failed: $($repository.Commit)" }
}

Write-Host "Pinned source trees are ready under $sourceRoot"
