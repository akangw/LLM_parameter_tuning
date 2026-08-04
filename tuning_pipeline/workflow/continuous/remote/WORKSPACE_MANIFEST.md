# vllmTKB current-version GLM-5.2 tuning workspace

This directory is the server-side execution and artifact workspace controlled
by the checkout containing this manifest. No absolute local checkout path is
part of the runtime contract.

## Safety boundary

- Writable root:
  `/mnt/host-model/slai/user-1-wangakang/wangakang/cjx-workspace/vllmtkb-418bd627-32c8cf190`
- Never run `rm` or another file-deletion operation.
- Shared model checkpoints and the parameter portrait are read-only inputs.
- Experiment artifacts are retained under `workflow/auto/runs` and
  `workflow/auto/lab_runs`.

## Fixed runtime inputs

- Runtime image digest:
  `sha256:37f43036c21c80c9cc6c6656f472df0f1b79d2fce2027168d270a43d725e0305`
- Benchmark image digest:
  `sha256:46d8bbb49f90f0607a41f972b7d847b434e0cf0f86fce9164549a7fdf0033112`
- Model:
  `/models/share/GLM-5.2-w8a8`
- Sliced MTP draft:
  `/models/share/GLM-5.2-w8a8-mtp-only-vllm0.25.1-image46d8bbb4`
- Deployment: two nodes, 16 NPUs per node, TP=16, DP=2.
- Default benchmark: read-only central `tuning-fixed` L1 v3 frozen matrix
  (1024/256, 8192/512, 1024/1024, 256/2048 at C1/C16/C32).
- Legacy optional benchmark: random 32K-centered input / 1K-centered output.

## Ownership of artifacts

- Local Windows project: parameter knowledge base, controller state, complete
  round archives, Codex decisions, and orchestration code.
- This server workspace: synchronized runtime scripts, lease templates,
  candidate files, raw logs, complete ServeBench results, consolidated
  benchmark outputs, and metrics.

The image is paired with a parameter portrait built from vLLM
`418bd6273c03bf48d5066733769e0a74bdc51694` and vllm-ascend
`32c8cf190f596b47f0d0b965e64aea9f2b789ad4`, matching the image package
metadata approved by the owner.
