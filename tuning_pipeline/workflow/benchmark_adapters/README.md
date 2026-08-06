# Custom benchmark adapter contract

Select `custom_adapter_v1`, then set `benchmark.custom_benchmark.adapter_path` to
a Python file below an `allowlisted_roots` directory. The Controller freezes and
copies that file during remote preparation. The tuning Agent cannot supply or
change the command.

The runner calls:

```text
python ADAPTER.py --request request.json --output adapter_output.json
```

The request contains `endpoint`, `served_model`, `run_dir`, and the frozen
`config` object. The output must contain a `metrics` object with these finite,
non-negative numeric fields:

- `successful_requests`
- `failed_requests`
- `output_token_throughput`
- `mean_ttft` (milliseconds)
- `mean_tpot` (milliseconds per output token)

The runner validates these fields and adds the frozen benchmark identity before
publishing `metrics.json`. See `example_http_adapter.py` for a stdlib-only
reference implementation. It is illustrative, not a recommended performance
method; the public vLLM profile is the portable default.
