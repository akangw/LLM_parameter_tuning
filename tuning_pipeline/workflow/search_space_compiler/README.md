# Independent Search-Space Compiler

This module turns tagged parameter knowledge into an auditable
`search_limits` proposal. It is intentionally isolated from the running
continuous controller.

## Safety boundary

- Reads local YAML/JSON evidence only.
- Does not use SSH, network calls, subprocesses, or controller imports.
- Does not edit `workflow/continuous/config.yaml`.
- Refuses to write output anywhere below `workflow/continuous`.
- Does not itself execute a proposal. The separately tested continuous
  adapter consumes it only while creating a new Session.

## Pipeline

```text
tag recall
  -> canonical names and explicit aliases
  -> scenario prerequisites
  -> current-image capability status
  -> injection-contract check
  -> fixed / tunable split
  -> baseline + registry + safe knowledge values
  -> machine-constraint pruning
  -> risk and approval classification
  -> 10-15 active parameters + reserve pool
  -> compiled proposal, audit, approval queue, manifest
```

Recall uses AND between tag dimensions and OR within one dimension. Matching is
exact: the tag `a3` does not accidentally match `a30`.

The active set is a per-Session tuning budget, not a permanent deletion of
other candidates. The default policy selects 13 diverse, high-impact
parameters and leaves the remaining eligible parameters in a reserve pool.
Future Sessions can rotate reserves into the active set based on prior results.
The compiled summary also reports the naive Cartesian-product size. The
compiler does not enumerate that product; a future optimizer should sample it
and call `validate_candidate` for every proposed joint configuration.

## History-aware Session rotation

History-aware selection is enabled only when `--history` is supplied. It
supports both the existing `history_input.json` list and a normalized object
with a `trials` list.

For every attributable parameter change it records:

- objective gain using `total_token_throughput`;
- failure rate for explicitly parameter-caused failures;
- median TTFT/TPOT guardrail regressions;
- gain variance as a stability signal;
- tested-value coverage and an exploration bonus;
- attribution confidence, reduced when multiple parameters changed together.

Failures classified as infrastructure or network failures are not charged to a
parameter. Only the parameter failure classifications allowed by `policy.yaml`
are counted. Explicit `parameter_invalid` and `parameter_oom` evidence
quarantines the failed value rather than automatically discarding the entire
parameter dimension.

Rotation is deliberately bounded:

- no attributable history means no rotation;
- at most two parameters are swapped per Session by default;
- an incoming Reserve must beat an outgoing parameter by the configured score
  margin;
- at least five core parameters remain active;
- every swap contains the incoming/outgoing scores and reasons.

## Inputs

- `scenario.glm52-a3-aligned-l1.yaml`: immutable scenario snapshot, baseline,
  topology, image identity, and verified capability evidence.
- `registry.yaml`: canonical names, aliases, roles, safe discrete values,
  prerequisites, risks, and proposed injection contracts.
- `policy.yaml`: recall, scoring, approval, and activation-budget policy.
- `tag_params/output/params`: existing tagged parameter knowledge.

A planned parameter that is absent from
`capabilities.verified_canonical_parameters` remains selectable for scientific
review, but is marked `human_required`; it cannot be auto-approved merely
because its declared risk is low.

Knowledge-base `suggested_values` are opt-in per registry entry. Values that
cannot be parsed as discrete scalars are ignored, and all accepted values still
pass the static and combination constraints. This permits program-derived
values without allowing free-form advice text to become an executable setting.

## Run

From the project root:

```powershell
python -m workflow.search_space_compiler --dry-run
```

The command above writes nothing. To save a standalone evidence package:

```powershell
python -m workflow.search_space_compiler `
  --output workflow/search_space_compiler/runs/my_review
```

To score a completed Session and perform bounded automatic rotation:

```powershell
python -m workflow.search_space_compiler `
  --history path/to/history_input.json `
  --previous-selection path/to/previous/search_space.compiled.yaml `
  --output workflow/search_space_compiler/runs/next_session_review
```

`--previous-selection` is optional for the first history-aware compilation. If
omitted, the cold-start 13-parameter selection is used as the previous set.

The output directory must not already exist. It contains:

- `search_space.compiled.yaml`: active limits and complete parameter records.
- `audit.json`: recalled, fixed, rejected, active, and reserve evidence.
- `approval_queue.yaml`: active parameters requiring human review.
- `rotation_report.yaml`: parameter statistics, quarantined values, and
  auditable swap reasons.
- `manifest.json`: hashes of all generated artifacts.

## Tests

```powershell
python -m unittest workflow.search_space_compiler.test_compiler -v
```

The tests are offline and use temporary output directories. They do not touch
the running task or controller configuration.

## Main-flow integration boundary

The adapter is implemented separately in
`workflow/continuous/search_space_adapter.py`. It:

1. runs only before a new Session is created;
2. accepts only explicitly approved planned parameters;
3. merges fixed runtime-contract fields without counting them as active;
4. freezes the compiled result and rotation report inside the Session archive;
5. leaves this compiler free of controller, SSH, submission, and remote-write
   dependencies.

Manual mode remains available in `workflow/continuous/config.yaml`. Resuming an
old Session loads its frozen configuration and never recompiles its limits.
