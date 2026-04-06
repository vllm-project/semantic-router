# TD003: Package Topology, Naming, and Hotspot Layout Debt

## Status

Closed

## Scope

code organization and file/module structure

## Summary

The broad package-topology debt that originally justified this umbrella entry has now been retired into stable subsystem seams and narrower follow-up debts. The original evidence files are no longer acting as catch-all owners for unrelated responsibilities:

- `src/semantic-router/pkg/config/config.go` is back to being the central schema table, with canonical conversion, plugin-family contracts, and validation split into adjacent files as part of TD026.
- `src/semantic-router/pkg/extproc/processor_req_body.go` and `processor_res_body.go` now sit on request/response phase seams with the extracted helpers introduced while closing TD023 and TD029.
- `src/vllm-sr/cli/core.py` is now a startup/shutdown orchestrator, while `src/vllm-sr/cli/runtime_status.py` owns local status/log probes and command shaping, and `src/vllm-sr/cli/commands/runtime.py` remains the Click entrypoint seam.
- dashboard builder/chat hotspots now have explicit sibling support modules and local rules that keep page/container orchestration separate from display fragments and helper logic.

Remaining oversized-file ratchets are still real debt, but they now belong to the structure-ratchet inventory in TD006 and the nearest local `AGENTS.md` files rather than to one repo-wide package-topology umbrella.

## Evidence

- [src/semantic-router/pkg/config/config.go](../../../src/semantic-router/pkg/config/config.go)
- [src/semantic-router/pkg/config/AGENTS.md](../../../src/semantic-router/pkg/config/AGENTS.md)
- [src/semantic-router/pkg/extproc/processor_req_body.go](../../../src/semantic-router/pkg/extproc/processor_req_body.go)
- [src/semantic-router/pkg/extproc/processor_res_body.go](../../../src/semantic-router/pkg/extproc/processor_res_body.go)
- [src/semantic-router/pkg/extproc/AGENTS.md](../../../src/semantic-router/pkg/extproc/AGENTS.md)
- [src/vllm-sr/cli/core.py](../../../src/vllm-sr/cli/core.py)
- [src/vllm-sr/cli/runtime_status.py](../../../src/vllm-sr/cli/runtime_status.py)
- [src/vllm-sr/cli/commands/runtime.py](../../../src/vllm-sr/cli/commands/runtime.py)
- [src/vllm-sr/cli/AGENTS.md](../../../src/vllm-sr/cli/AGENTS.md)
- [dashboard/frontend/src/pages/BuilderPage.tsx](../../../dashboard/frontend/src/pages/BuilderPage.tsx)
- [dashboard/frontend/src/pages/AGENTS.md](../../../dashboard/frontend/src/pages/AGENTS.md)
- [dashboard/frontend/src/components/ChatComponent.tsx](../../../dashboard/frontend/src/components/ChatComponent.tsx)
- [dashboard/frontend/src/components/AGENTS.md](../../../dashboard/frontend/src/components/AGENTS.md)
- [tools/agent/structure-rules.yaml](../../../tools/agent/structure-rules.yaml)
- [docs/agent/tech-debt/td-006-structural-rule-target-vs-legacy-hotspots.md](td-006-structural-rule-target-vs-legacy-hotspots.md)

## Why It Matters

- The desired structure rules say files should stay narrow and packages should reflect clear seams, but the codebase still contains several oversized hotspots and a pkg layout that is partly too flat and partly too fragmented.
- Some packages carry only a tiny amount of code while other high-complexity areas are still concentrated in large orchestration files.
- Naming and package boundaries do not always reflect the current architectural layers.

## Desired End State

- Package boundaries reflect real subsystems and runtime seams.
- Legacy hotspot files continue shrinking until the main orchestration files stop acting as catch-all modules.

## Exit Criteria

- Satisfied on 2026-04-06: the highest-risk package-topology hotspots now sit behind stable subsystem or phase seams, and new work is guided by local rule files instead of one repo-wide “topology is broken” umbrella.
- Satisfied on 2026-04-06: package names and directory structure now align closely enough with stable subsystem boundaries that remaining oversized-file concerns are better tracked as structural-ratchet debt in TD006.

## Resolution

- `src/semantic-router/pkg/config/` now keeps schema declaration in `config.go` while canonical conversion/export, plugin-family contracts, and validation live in adjacent support files, which retired the config-side portion of this umbrella under TD026.
- `src/semantic-router/pkg/extproc/` now treats `processor_req_body.go` and `processor_res_body.go` as phase orchestrators with extracted request/response helpers and explicit local rules, which retired the extproc-side portion of this umbrella under TD023 and TD029.
- `src/vllm-sr/cli/core.py` now owns startup/shutdown orchestration only; this change extracted local `status` / `logs` probing and filtered log command assembly into `src/vllm-sr/cli/runtime_status.py`, and the CLI local rules plus repo map now point at that split explicitly.
- Dashboard builder/chat package seams now rely on sibling support modules and local `AGENTS.md` guidance to keep page/container orchestration separated from support logic; remaining size-only ratchets stay in TD006 rather than reopening this broad topology debt.

## Validation

- `python -m compileall /Users/bitliu/vs/src/vllm-sr/cli/core.py /Users/bitliu/vs/src/vllm-sr/cli/runtime_status.py`
