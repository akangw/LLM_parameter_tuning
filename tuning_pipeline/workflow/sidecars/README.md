# Offline Tuning Sidecars

This directory contains two modules that remain independently executable. New
continuous Sessions may also consume them when `sidecars.enabled: true`.
Neither module itself submits or modifies remote jobs.

## Safety boundary

- Local YAML/JSON reads only.
- No SSH, network requests, subprocesses, or remote writes.
- No import from `workflow/continuous`.
- Standalone CLI use rejects output/state paths below `workflow/continuous`.
- The Controller must explicitly opt into a Session-scoped rule store; that
  capability is not exposed by the CLI.
- Existing output evidence files are never overwritten.
- No current Session configuration or task state is changed.

## Portrait retriever

`portrait_retriever.py` retrieves the full natural-language portraits for the
parameters an Agent plans to change. It then follows `related_parameters` for
one hop. All duplicate/alias portrait variants are retained for audit instead
of silently selecting one.

The optional Search Limits input is a boundary: a requested changed parameter
outside that Session's limits is rejected.

Dry run to stdout:

```powershell
python -m workflow.sidecars.portrait_retriever `
  --parameter compilation_enable_sp `
  --search-limits path/to/search_space.compiled.yaml `
  --scenario workflow/search_space_compiler/scenario.glm52-a3-aligned-l1.yaml
```

Write a new standalone evidence file:

```powershell
python -m workflow.sidecars.portrait_retriever `
  --parameter compilation_enable_sp `
  --output workflow/sidecars/runs/sp-review.yaml
```

The evidence keeps `constraints`, `related_parameters`, `valid_choices`,
`tuning_advice`, source locations, and all other existing portrait fields
verbatim. It does not convert natural language into executable rules.

## Runtime rule store

`runtime_rule_store.py` maintains a deterministic final-gate rule store. The
initial rules mirror already-established compiler/controller contracts. In an
integrated Session, a frozen copy is evaluated before every submission.

Create a new store:

```powershell
python -m workflow.sidecars.runtime_rule_store init `
  --store workflow/sidecars/runs/review-rules.yaml
```

Evaluate a candidate without executing it:

```powershell
python -m workflow.sidecars.runtime_rule_store evaluate `
  --store workflow/sidecars/runs/review-rules.yaml `
  --candidate path/to/candidate.yaml `
  --scenario workflow/search_space_compiler/scenario.glm52-a3-aligned-l1.yaml `
  --search-limits path/to/search_limits.yaml
```

Ingest completed history:

```powershell
python -m workflow.sidecars.runtime_rule_store ingest-history `
  --store workflow/sidecars/runs/review-rules.yaml `
  --history path/to/history_input.json `
  --scenario workflow/search_space_compiler/scenario.glm52-a3-aligned-l1.yaml
```

History feedback is conservative:

- a single parameter explicitly classified `parameter_invalid` or
  `parameter_oom` automatically quarantines only that exact value and scope;
- multi-parameter or generalized failures create `proposed` rules;
- proposed/shadow rules only warn;
- a proposal becomes blocking only after an explicit transition to `active`;
- infrastructure/network failures do not create rules.

Explicit proposal lifecycle:

```powershell
python -m workflow.sidecars.runtime_rule_store transition `
  --store workflow/sidecars/runs/review-rules.yaml `
  --proposal-id proposal:forbidden_combination:... `
  --status shadow
```

## Tests

```powershell
python -m unittest workflow.sidecars.test_sidecars -v
```

The tests use temporary directories and do not touch the running controller,
remote server, or current Session.

## Integrated Session lifecycle

With `sidecars.enabled: true`, Session creation freezes:

- full portraits for all active parameters and their one-hop relations;
- a compact portrait context embedded in every Agent analysis;
- an independent runtime rule store.

After an Agent selects changes, the exact changed-parameter portrait recall is
archived in that round. Candidate submission is blocked on active rules and
exact-value quarantines. History updates may add quarantines or proposals, but
never auto-activate a generalized rule.
