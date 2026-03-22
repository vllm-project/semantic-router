# TD035: Projection Partition Default Coverage Contract Is No Longer Declarative Only

## Status

Closed

## Scope

`spec/dsl.md`, `src/semantic-router/pkg/{dsl,classification,config}/**`, and targeted tests or docs that define or enforce projection-partition fallback behavior

## Summary

The repository now parses, compiles, decompiles, validates, and enforces `default` on `PROJECTION partition` as a live runtime fallback contract. The DSL authoring surface and the canonical runtime/config surface now both describe those partitions under `routing.projections.partitions`. Request-time grouped signal evaluation synthesizes the declared default member when no member in the partition fires, and native DSL validation also surfaces centroid-similarity warnings for `softmax_exclusive` embedding partitions. The previous declarative-only gap is retired.

## Evidence

- [spec/dsl.md](../../../spec/dsl.md)
- [src/semantic-router/pkg/config/projection_config.go](../../../src/semantic-router/pkg/config/projection_config.go)
- [src/semantic-router/pkg/dsl/validator_conflicts.go](../../../src/semantic-router/pkg/dsl/validator_conflicts.go)
- [src/semantic-router/pkg/dsl/dsl_test.go](../../../src/semantic-router/pkg/dsl/dsl_test.go)
- [src/semantic-router/pkg/classification/classifier_signal_groups.go](../../../src/semantic-router/pkg/classification/classifier_signal_groups.go)
- [src/semantic-router/pkg/classification/classifier_signal_groups_test.go](../../../src/semantic-router/pkg/classification/classifier_signal_groups_test.go)

## Why It Matters

- `default` must mean the same thing across the DSL, compiled config, and request-time routing. If any layer drifts, authors get a false sense of coverage.
- Native validation for `softmax_exclusive` partitions needs to catch ambiguous embedding centroids before deployment; otherwise a config can look structurally valid while still producing near-uniform softmax winners.

## Desired End State

- `PROJECTION partition.default` has one explicit, contributor-visible runtime meaning that matches the DSL contract.
- Native DSL validation warns when `softmax_exclusive` embedding-partition centroids are too similar to produce stable winners.
- Unit coverage demonstrates partition fallback behavior and centroid warning behavior.

## Exit Criteria

- Satisfied on 2026-03-21: request-time grouped signal evaluation synthesizes the declared default member when no member in the partition matches.
- Satisfied on 2026-03-21: native DSL validation warns when `softmax_exclusive` embedding-partition centroids exceed the ambiguity threshold.
- Satisfied on 2026-03-21: tests cover partition fallback behavior and centroid warning analysis so the contract cannot silently drift back to declarative-only behavior.

## Resolution

- `src/semantic-router/pkg/classification/classifier_signal_groups.go` now appends the declared partition default when no member in the partition fires, so grouped routes retain a live fallback contract at request time under `routing.projections.partitions`.
- `src/semantic-router/pkg/classification/classifier_signal_group_similarity.go` computes embedding-rule centroids and reports near-uniform `softmax_exclusive` groups through native validation.
- `src/semantic-router/cmd/dsl/test_runner.go` and `src/semantic-router/pkg/dsl/cli.go` route those native warnings into `sr-dsl validate`, while browser/WASM validation keeps an explicit warning that those runtime checks require native execution.
