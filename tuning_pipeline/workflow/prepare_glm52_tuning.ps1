[CmdletBinding()]
param(
    [string]$RemoteHost = "hetao-npu",
    [string]$RemoteProject = "/mnt/host-model/slai/user-1-wangakang/wangakang/cjx-workspace",
    [string]$RemoteModel = "/models/share/GLM-5.2-w8a8",
    [string]$KnowledgeBase = "",
    [ValidateSet("throughput", "ttft", "tpot", "memory")]
    [string]$OptimizeTarget = "throughput",
    [ValidateSet("long_input", "long_output", "high_concurrency")]
    [string]$DeployScenario = "long_input",
    [string]$OutputRoot = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($KnowledgeBase)) { $KnowledgeBase = Split-Path -Parent $PSScriptRoot }
if ([string]::IsNullOrWhiteSpace($OutputRoot)) { $OutputRoot = Join-Path $PSScriptRoot "runs" }

function Invoke-RemoteRead {
    param([Parameter(Mandatory)][string]$Command)
    # Fixed commands below are read-only. This server may return a nonzero SSH
    # status after successfully emitting stdout, so accept stdout when present.
    $previousErrorPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { $result = & ssh -o BatchMode=yes -o ConnectTimeout=15 $RemoteHost $Command 2>$null | Out-String }
    finally { $ErrorActionPreference = $previousErrorPreference }
    if ([string]::IsNullOrWhiteSpace($result)) { throw "Remote read returned no data: $Command" }
    return $result.TrimEnd()
}

if (-not (Test-Path -LiteralPath (Join-Path $KnowledgeBase "query.py"))) {
    throw "Knowledge base query.py was not found: $KnowledgeBase"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = Join-Path $OutputRoot "glm52_tuning_discovery_$timestamp"
New-Item -ItemType Directory -Path $runDir -Force | Out-Null

$modelText = Invoke-RemoteRead "sed -n '1,500p' '$RemoteModel/config.json'"
$readmeText = Invoke-RemoteRead "cd '$RemoteProject' && sed -n '1,260p' README.md"
$node0Text = Invoke-RemoteRead "cd '$RemoteProject' && sed -n '1,300p' node0_co.sh"
$node1Text = Invoke-RemoteRead "cd '$RemoteProject' && sed -n '1,300p' node1_co.sh"
$benchmarkText = Invoke-RemoteRead "cd '$RemoteProject' && sed -n '1,220p' vllm-benchmarking.sh"
$runtimeText = Invoke-RemoteRead "printf '%s\\n' '--- npu-smi ---'; npu-smi info; printf '%s\\n' '--- network ---'; ip -brief address; printf '%s\\n' '--- versions ---'; vllm --version; python --version"

$modelText | Set-Content -LiteralPath (Join-Path $runDir "model_config.json") -Encoding utf8
@("===== README.md =====", $readmeText, "", "===== node0_co.sh =====", $node0Text, "", "===== node1_co.sh =====", $node1Text, "", "===== vllm-benchmarking.sh =====", $benchmarkText) |
    Set-Content -LiteralPath (Join-Path $runDir "deployment_snapshot.txt") -Encoding utf8
$runtimeText | Set-Content -LiteralPath (Join-Path $runDir "runtime_snapshot.txt") -Encoding utf8

$model = $modelText | ConvertFrom-Json
$modelTags = [System.Collections.Generic.List[string]]::new()
if (($model.PSObject.Properties.Name -contains "n_routed_experts" -and [int]$model.n_routed_experts -gt 0) -or "$($model.model_type)" -match "moe" -or (@($model.architectures) -join " ") -match "Moe|MoE") { $modelTags.Add("moe") }
if (($model.PSObject.Properties.Name -contains "quantize") -or ($model.PSObject.Properties.Name -contains "quantization_config")) { $modelTags.Add("quantized") }
if ($modelTags.Count -eq 0) { throw "Could not infer a supported model tag from the model config." }

$hardwareTag = if ($readmeText -match "A3") { "a3" } else { throw "Cannot infer hardware=a3 from the deployment README." }
$topologyTag = if ($node0Text -match "--data-parallel-size 2" -and $node1Text -match "--headless") { "multi_node" } else { throw "Cannot infer a multi-node deployment from node scripts." }
$tagArgs = @("--tag", "hardware=$hardwareTag")
foreach ($tag in $modelTags) { $tagArgs += @("--tag", "model=$tag") }
$tagArgs += @("--tag", "deploy_topology=$topologyTag", "--tag", "deploy_scenario=$DeployScenario", "--tag", "optimize_target=$OptimizeTarget", "--where", "performance_impact=high", "--show", "name,valid_choices,tuning_advice.suggested_values,constraints,category", "--format", "yaml")

Push-Location $KnowledgeBase
try { $queryOutput = & python ".\query.py" @tagArgs 2>$null | Out-String }
finally { Pop-Location }
if ([string]::IsNullOrWhiteSpace($queryOutput)) { throw "The tag query returned no parameters." }
$queryOutput.TrimEnd() | Set-Content -LiteralPath (Join-Path $runDir "glm5.2_search_space.yaml") -Encoding utf8

$fingerprint = [ordered]@{
    model_path = $RemoteModel; architectures = @($model.architectures); model_type = $model.model_type
    routed_experts = $model.n_routed_experts; mtp_prediction_layers = $model.num_nextn_predict_layers
    attention_heads = $model.num_attention_heads; key_value_heads = $model.num_key_value_heads
    inferred_tags = [ordered]@{ hardware = $hardwareTag; model = @($modelTags); deploy_topology = $topologyTag; deploy_scenario = $DeployScenario; optimize_target = $OptimizeTarget }
    fixed_benchmark = [ordered]@{ input_tokens = 32000; output_tokens = 1000; prompts = 8; request_rate = 0.2; temperature = 0 }
    source_artifacts = [ordered]@{ model_config = "model_config.json"; deployment_snapshot = "deployment_snapshot.txt"; runtime_snapshot = "runtime_snapshot.txt"; search_space = "glm5.2_search_space.yaml" }
}
$fingerprint | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $runDir "tuning_fingerprint.json") -Encoding utf8

$agentContext = @"
# Codex tuning task context

Read tuning_fingerprint.json, deployment_snapshot.txt, runtime_snapshot.txt,
glm5.2_search_space.yaml, and $KnowledgeBase\SKILL-with-tag.md.

Goal: improve output throughput for the fixed 32K-input / 1K-output benchmark.

Rules:
1. Treat the deployment snapshot as the baseline and propose no more than one startup-configuration change in the first experiment.
2. For every recommended parameter, run query.py with --show-all and cite constraints, suggested_values, and the baseline value.
3. Do not change model path, DP/TP, IPs, network settings, ports, or execute any remote command. This phase is planning only.
4. Exclude PD-disaggregation, LoRA, and KV-transfer settings unless their prerequisites already exist in the deployment snapshot.
5. Report an exact before/after launch-flag diff and benchmark plan. Do not claim improvement before server testing.
"@
$agentContext | Set-Content -LiteralPath (Join-Path $runDir "codex_agent_context.md") -Encoding utf8

Write-Host "Discovery complete: $runDir"
