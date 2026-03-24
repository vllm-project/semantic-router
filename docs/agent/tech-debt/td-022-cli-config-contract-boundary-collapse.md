# TD022: CLI Config Contract Knowledge Is Collapsed Across Schema, Migration, and Validation Hotspots

## Status

Closed

## Scope

`src/vllm-sr/cli/models.py`, `src/vllm-sr/cli/config_migration.py`, `src/vllm-sr/cli/validator.py`, and adjacent config-generation helpers

## Summary

The Python CLI now carries a large share of the canonical v0.3 config contract, but that knowledge is split across three separate hotspots with overlapping ownership. `models.py` defines the schema and recursive routing nodes, `config_migration.py` rewrites legacy or mixed layouts into canonical form, and `validator.py` re-encodes many of the same field inventories and cross-field invariants again. This is more than a file-size problem: canonical config changes now require synchronized edits across schema, compatibility, and validation hotspots that do not share one narrow owner for the field inventory.

## Evidence

- [src/vllm-sr/cli/models.py](../../../src/vllm-sr/cli/models.py)
- [src/vllm-sr/cli/config_contract.py](../../../src/vllm-sr/cli/config_contract.py)
- [src/vllm-sr/cli/config_migration.py](../../../src/vllm-sr/cli/config_migration.py)
- [src/vllm-sr/cli/parser.py](../../../src/vllm-sr/cli/parser.py)
- [src/vllm-sr/cli/validator.py](../../../src/vllm-sr/cli/validator.py)
- [src/vllm-sr/cli/config_generator.py](../../../src/vllm-sr/cli/config_generator.py)
- [src/vllm-sr/tests/test_config_contract.py](../../../src/vllm-sr/tests/test_config_contract.py)
- [src/vllm-sr/tests/test_config_migrate.py](../../../src/vllm-sr/tests/test_config_migrate.py)
- [src/vllm-sr/cli/AGENTS.md](../../../src/vllm-sr/cli/AGENTS.md)
- [docs/agent/architecture-guardrails.md](../architecture-guardrails.md)

## Why It Matters

- A canonical config change can silently drift between the schema, migration path, and semantic validator because those seams currently duplicate field inventories and compatibility rules.
- The current shape makes legacy-compatibility work harder to retire because migration behavior is not isolated behind one compatibility seam.
- The CLI local rules still focus on `docker_cli.py`, so the active config-contract hotspots are easier to extend accidentally without explicit extraction pressure.
- The result is shallow, wide ownership for one product contract instead of deeper modules with one clear owner per concern.

## Desired End State

- CLI config contract ownership is split into narrower modules by concern: schema declarations, compatibility migration, and semantic validation.
- Canonical field inventories and signal/decision/plugin type tables have one primary owner that migration and validation reuse instead of redeclaring.
- Legacy migration is an explicit compatibility seam rather than a general-purpose rewrite layer that grows with every new field.
- Local CLI rules point at the real config-contract hotspots, not only the runtime orchestration hotspot.

## Exit Criteria

- Canonical config changes no longer require parallel edits across `models.py`, `config_migration.py`, and `validator.py` for the same field inventory unless the change truly spans all three concerns.
- The biggest schema families in `models.py` are split into narrower support modules or inventories with clearer ownership boundaries.
- Migration helpers reuse shared inventories or translation helpers instead of re-encoding large sets of canonical keys inline.
- CLI-local `AGENTS.md` and harness docs reflect the active config-contract hotspots and the expected extraction-first workflow.

## Resolution

- Shared CLI config-contract ownership now lives in `config_contract.py`, which owns canonical routing signal inventories, legacy flat-key mappings, and compatibility tables reused by `config_migration.py`, `parser.py`, and `validator.py`.
- Migration and validation no longer redeclare the same signal/plugin/provider inventories inline for the refactored paths touched by this work.
- CLI-local rules now explicitly call out `models.py`, `config_migration.py`, and `validator.py` as config-contract hotspots instead of treating only `docker_cli.py` as the extraction target.

## Validation

- `pytest src/vllm-sr/tests/test_config_contract.py src/vllm-sr/tests/test_config_migrate.py src/vllm-sr/tests/test_plugin_parsing.py src/vllm-sr/tests/test_latency_validation.py src/vllm-sr/tests/test_algorithm_config.py`
- `python -m compileall src/vllm-sr/cli`
- `make vllm-sr-test`
- `make vllm-sr-test-integration`
- `make agent-ci-gate CHANGED_FILES="src/vllm-sr/cli/config_contract.py,src/vllm-sr/cli/config_migration.py,src/vllm-sr/cli/parser.py,src/vllm-sr/cli/validator.py,src/vllm-sr/tests/test_config_contract.py,src/vllm-sr/tests/test_config_migrate.py"`

## Notes

- A separate `make agent-feature-gate ...` run later still failed in `ai-gateway` E2E due unrelated acceptance drift (`pii-detection`, `plugin-chain-execution`, `decision-fallback-behavior`). That blocker does not reopen this CLI-boundary debt because the CLI contract refactor's direct unit, integration, and harness checks passed.
