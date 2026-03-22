# TD035: SIGNAL_GROUP Default Coverage Contract Is Still Declarative Only

## Status

Closed

## Scope

`spec/dsl.md`, `src/semantic-router/pkg/{dsl,classification,config}/**`, and targeted tests or docs that define or enforce `SIGNAL_GROUP`

## Summary

The repository now parses, compiles, decompiles, validates, and enforces `SIGNAL_GROUP.default` as a live runtime fallback contract. The DSL authoring surface still uses `SIGNAL_GROUP`, while the canonical runtime/config surface now stores those partitions under `routing.projections.partitions`. Request-time grouped signal evaluation synthesizes the declared default member when no member in the group fires, and native DSL validation also surfaces centroid-similarity warnings for `softmax_exclusive` embedding groups. The previous declarative-only gap is retired.

## Evidence

- [spec/dsl.md](../../../spec/dsl.md)
- [src/semantic-router/pkg/config/projection_config.go](../../../src/semantic-router/pkg/config/projection_config.go)
- [src/semantic-router/pkg/dsl/validator_conflicts.go](../../../src/semantic-router/pkg/dsl/validator_conflicts.go)
- [src/semantic-router/pkg/dsl/dsl_test.go](../../../src/semantic-router/pkg/dsl/dsl_test.go)
- [src/semantic-router/pkg/classification/classifier_signal_groups.go](../../../src/semantic-router/pkg/classification/classifier_signal_groups.go)
- [src/semantic-router/pkg/classification/classifier_signal_groups_test.go](../../../src/semantic-router/pkg/classification/classifier_signal_groups_test.go)

## Why It Matters

- `default` must mean the same thing across the DSL, compiled config, and request-time routing. If any layer drifts, authors get a false sense of coverage.
- Native validation for `softmax_exclusive` groups needs to catch ambiguous embedding centroids before deployment; otherwise a config can look structurally valid while still producing near-uniform softmax winners.

## Desired End State

- `SIGNAL_GROUP.default` has one explicit, contributor-visible runtime meaning that matches the DSL contract.
- Native DSL validation warns when `softmax_exclusive` embedding-group centroids are too similar to produce stable winners.
- Unit coverage demonstrates grouped-route fallback behavior and centroid warning behavior.

## Exit Criteria

- Satisfied on 2026-03-21: request-time grouped signal evaluation synthesizes the declared default member when no member in the group matches.
- Satisfied on 2026-03-21: native DSL validation warns when `softmax_exclusive` embedding-group centroids exceed the ambiguity threshold.
- Satisfied on 2026-03-21: tests cover grouped-route fallback behavior and centroid warning analysis so the contract cannot silently drift back to declarative-only behavior.

## Resolution

- `src/semantic-router/pkg/classification/classifier_signal_groups.go` now appends the declared group default when no member in the group fires, so grouped routes retain a live fallback contract at request time, even though the canonical config surface now stores those groups under `routing.projections.partitions`.
- `src/semantic-router/pkg/classification/classifier_signal_group_similarity.go` computes embedding-rule centroids and reports near-uniform `softmax_exclusive` groups through native validation.
- `src/semantic-router/cmd/dsl/test_runner.go` and `src/semantic-router/pkg/dsl/cli.go` route those native warnings into `sr-dsl validate`, while browser/WASM validation keeps an explicit warning that those runtime checks require native execution.
