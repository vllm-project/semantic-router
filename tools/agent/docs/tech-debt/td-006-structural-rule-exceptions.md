# TD006: Structural Rule Exceptions Still Cover Active Code

## Status

Open

## Owner Plan

PL0032 Architecture Debt Consolidation

## Release Relevance

None - non-release debt

## Scope

Repo-wide function-size, nesting, and interface-size exceptions in the agent
structure gate.

## Summary

The intended architecture is simple: most changed source files should pass the
shared AST structure limits directly. File length is advisory because it cannot
distinguish a cohesive deep module from arbitrary fragmentation. The current
source tree still needs explicit function, nesting, or interface exceptions in
router, operator, dashboard, training, CLI, and native-binding surfaces.

The debt is not the existence of the gate. The debt is that too many maintained
files still need `function_checks` or `interface_checks` relaxation, or a
baseline ratchet, in
[tools/agent/structure-rules.yaml](../../../../tools/agent/structure-rules.yaml).
Those exceptions keep validation usable, but they also show where executable
control flow or interfaces still do not match the target shape.

## Evidence

- [tools/agent/structure-rules.yaml](../../../../tools/agent/structure-rules.yaml)
- [tools/agent/scripts/structure_check.py](../../../../tools/agent/scripts/structure_check.py)
- [tools/agent/docs/architecture-guardrails.md](../architecture-guardrails.md)
- [tools/agent/docs/repo-map.md](../repo-map.md)
- [tools/linter/go/.golangci.agent.yml](../../../../tools/linter/go/.golangci.agent.yml)

Current exception groups include:

- router config, DSL, classifier, extproc, cache, response-store, looper,
  selection, modelselection, tools, and memory helpers
- operator API/controller schema and regression-test surfaces
- dashboard frontend config/page/components and backend control handlers
- training verification scripts and native config-loading bindings
- CLI config generation, migration, parser, and model schema surfaces

## Why It Matters

- Structural exceptions are useful as a ratchet, but every exception widens the
  set of files where small follow-up changes can avoid the standard shape.
- Broad orchestrators make release work harder when they mix schema,
  validation, runtime orchestration, and transport logic in one place.
- Maintainers need the exception list to read as a current target list, not as a
  history of past migrations.

## Desired End State

- The structure gate's default AST limits are the common case for changed
  source files, while file length remains review evidence.
- Remaining exceptions are rare, clearly justified, and owned by one active TD
  or release plan.
- Router, dashboard, operator, CLI, training, and binding changes land through
  narrower helpers instead of reopening broad orchestrator files.

## Exit Criteria

- `tools/agent/structure-rules.yaml` has no broad exception groups for actively
  maintained source files.
- Go agent lint exclusions and AST structure-rule exceptions agree on the same
  small set of files, or both can be removed for a surface.
- New work can pass changed-file lint and structure checks without adding
  another exception for the touched area.
