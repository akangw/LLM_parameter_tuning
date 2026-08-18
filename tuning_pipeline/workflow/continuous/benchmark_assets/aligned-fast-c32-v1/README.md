# Aligned Fast C32 v1

This is the repository-owned overlay for the fixed, sub-10-minute benchmark.
The server asset is created by copying `benchmark-tuning-structured-v4` into a
new `benchmark-tuning-fast-c32-v1` directory and then adding the suite in this
overlay. The original v4 directory is never modified.

The fast suite preserves all four frozen workloads and all 64 formal C32
requests per workload. It removes only C1/C16 and therefore retains the exact
primary-score cases used by the previous full matrix. The measured A8 formal
C32 duration was 394.49 seconds; the hard execution budget is 600 seconds.
