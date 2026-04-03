# TD003: Package Topology, Naming, and Hotspot Layout Debt

## Status

Open

## Scope

code organization and file/module structure

## Summary

The codebase still contains oversized hotspots and uneven package seams that do not yet reflect the intended subsystem boundaries.

## Evidence

- [src/semantic-router/pkg/config/config.go](../../../src/semantic-router/pkg/config/config.go)
- [src/semantic-router/pkg/extproc/processor_req_body.go](../../../src/semantic-router/pkg/extproc/processor_req_body.go)
- [src/semantic-router/pkg/extproc/processor_res_body.go](../../../src/semantic-router/pkg/extproc/processor_res_body.go)
- [src/vllm-sr/cli/core.py](../../../src/vllm-sr/cli/core.py)
- [src/vllm-sr/cli/commands/runtime.py](../../../src/vllm-sr/cli/commands/runtime.py)
- [dashboard/frontend/src/pages/BuilderPage.tsx](../../../dashboard/frontend/src/pages/BuilderPage.tsx)
- [dashboard/frontend/src/components/ChatComponent.tsx](../../../dashboard/frontend/src/components/ChatComponent.tsx)

## Why It Matters

- The desired structure rules say files should stay narrow and packages should reflect clear seams, but the codebase still contains several oversized hotspots and a pkg layout that is partly too flat and partly too fragmented.
- Some packages carry only a tiny amount of code while other high-complexity areas are still concentrated in large orchestration files.
- Naming and package boundaries do not always reflect the current architectural layers.

## Desired End State

- Package boundaries reflect real subsystems and runtime seams.
- Legacy hotspot files continue shrinking until the main orchestration files stop acting as catch-all modules.

## Exit Criteria

- The highest-risk hotspots have been reduced enough that new work can follow the standard modularity rules without exception-heavy local guidance.
- Package names and directory structure align more closely with stable subsystem boundaries.
