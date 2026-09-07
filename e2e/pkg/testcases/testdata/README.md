# Selector algorithm coverage manifest

`selector_algorithm_coverage.json` maps the selector entries in
`config.DecisionAlgorithmCatalog()` to their maintained E2E paths. It is the
issue #3179 rollout ledger, not a second runtime algorithm catalog.

Coverage states have these meanings:

- `planned`: the algorithm is accounted for, but it has no runnable contract
  yet. `target_profile` records where its follow-up should land.
- `partial`: the named testcase is runnable from the named profile, but does
  not yet prove the complete selected-model, diagnostics, and failure or
  fallback contract.
- `covered`: the named testcase and profile provide the complete deterministic
  contract required by issue #3179.

The config package test derives the selector set and tiers from the runtime
catalog and rejects missing, extra, duplicate, or tier-mismatched manifest
entries. The E2E package test rejects unknown states and ensures every
`partial` or `covered` testcase is registered and reachable from its profile.

An algorithm-specific PR should update one existing entry rather than add a
separate algorithm list. It may change `planned` to `partial` while bringing up
fixtures, but it should only use `covered` after the deterministic success,
diagnostic, and applicable failure or fallback assertions are all maintained.

The legacy `core-selection-*` registrations and their fixture remain during
the rollout. Retire them together after every manifest entry is `covered`, so
the migration never removes historical probes before their replacements land.
