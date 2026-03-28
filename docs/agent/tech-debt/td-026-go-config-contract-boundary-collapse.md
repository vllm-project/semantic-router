# TD026: Go Router Config Contract Knowledge Still Collapses Across Schema Families, Canonical Conversion, and Validation Hotspots

## Status

Closed

## Scope

`src/semantic-router/pkg/config/{config.go,validator.go,rag_plugin.go,canonical_*.go}` and adjacent config-contract tests or docs

## Summary

The Go router config package had concentrated too much contract knowledge into a small set of hotspots. `config.go` carried broad runtime schema families and inline module contracts, `validator.go` aggregated unrelated semantic checks, `rag_plugin.go` concentrated one growing plugin family's backend contracts and validation path, and canonical import/export or normalization behavior spanned `canonical_config.go`, `canonical_export.go`, and related helpers. This was more than a file-size issue: one config-contract change could require synchronized edits across schema inventories, canonical conversion, plugin-family payloads, and semantic validation seams. TD006 already tracked structural hotspot debt, but the repo lacked a subsystem-specific debt record for this runtime config contract collapse.

`validator.go` is now reduced to the package entrypoint, with family-specific checks moved into `validator_decision.go`, `validator_modality.go`, and `validator_tool_filtering.go`. The RAG plugin family is likewise split across `rag_plugin.go`, `rag_plugin_backends.go`, and `rag_plugin_validation.go`, with dedicated tests in `rag_plugin_test.go`. Package tests, calling-site tests, harness validation, and CPU-local smoke are all green, so this debt no longer needs to stay open.

## Evidence

- [src/semantic-router/pkg/config/config.go](../../../src/semantic-router/pkg/config/config.go)
- [src/semantic-router/pkg/config/validator.go](../../../src/semantic-router/pkg/config/validator.go)
- [src/semantic-router/pkg/config/rag_plugin.go](../../../src/semantic-router/pkg/config/rag_plugin.go)
- [src/semantic-router/pkg/config/canonical_config.go](../../../src/semantic-router/pkg/config/canonical_config.go)
- [src/semantic-router/pkg/config/canonical_export.go](../../../src/semantic-router/pkg/config/canonical_export.go)
- [src/semantic-router/pkg/config/AGENTS.md](../../../src/semantic-router/pkg/config/AGENTS.md)
- [tools/agent/structure-rules.yaml](../../../tools/agent/structure-rules.yaml)
- [docs/agent/tech-debt/td-006-structural-rule-target-vs-legacy-hotspots.md](td-006-structural-rule-target-vs-legacy-hotspots.md)
- [docs/agent/architecture-guardrails.md](../architecture-guardrails.md)

## Why It Matters

- Canonical config changes can drift between runtime schema declarations, canonical normalization/export, plugin-family payloads, and semantic validation because ownership is still too broad.
- `validator.go` is at risk of remaining the default sink for every new semantic rule, which keeps the package shallow and hard to reason about.
- Plugin-family growth, especially in RAG, keeps adding backend-specific payloads and validation branches to one hotspot instead of growing behind deeper support modules.
- The package has local rules for `config.go`, but without a dedicated debt item the repo still lacks a durable plan for the broader contract boundary cleanup.

## Desired End State

- Runtime schema families, canonical normalization/export, plugin-family-specific contracts, and semantic validation each have clearer primary owners.
- `config.go` stays focused on shared runtime schema tables and central config contracts instead of reabsorbing conversion or plugin-family logic.
- `validator.go` becomes a thinner entrypoint that delegates family-specific checks to focused support files.
- Plugin-family hotspots such as RAG split backend-specific payloads, decode helpers, or validation logic into adjacent modules before they become second schema tables.

## Exit Criteria

- New runtime or canonical config features no longer require parallel edits across `config.go`, `validator.go`, and `rag_plugin.go` unless the change is truly cross-cutting.
- Canonical import/export and normalization stay behind dedicated support files rather than drifting back into `config.go`.
- Family-specific semantic checks are split enough that `validator.go` is no longer the default owner for every new rule.
- Config-local `AGENTS.md` and harness docs explicitly name the active hotspots and the extraction-first path contributors should follow.

## Resolution

- Semantic validation now fans out from `validator.go` into focused decision, modality, and tool-filtering validators instead of continuing to accumulate unrelated rule families in one file.
- The RAG plugin family now splits backend payload contracts and validation logic across `rag_plugin_backends.go` and `rag_plugin_validation.go`, with `rag_plugin.go` kept as the thinner entrypoint seam.
- Config-local rules and the shared architecture docs now explicitly treat schema tables, canonical conversion, plugin-family contracts, and semantic validation as separate owners inside `src/semantic-router/pkg/config`.

## Validation

- `go test ./pkg/config/...`
- `go test ./pkg/extproc/...`
- `make test-semantic-router`
- `make agent-validate`
- `make agent-ci-gate CHANGED_FILES="..."`
- `make agent-serve-local ENV=cpu AGENT_STACK_NAME=pl0008-smoke AGENT_PORT_OFFSET=400`
- `make agent-smoke-local AGENT_STACK_NAME=pl0008-smoke AGENT_PORT_OFFSET=400`
- `make agent-stop-local ENV=cpu AGENT_STACK_NAME=pl0008-smoke AGENT_PORT_OFFSET=400`
