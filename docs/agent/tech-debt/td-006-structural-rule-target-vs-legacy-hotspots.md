# TD006: Structural Rule Target Still Exceeds Reality in Key Legacy Hotspots

## Status

Closed

## Scope

The shared harness docs and executable rule layer that own how oversized legacy hotspots are discovered, routed, and constrained across router-core, dashboard, operator, training, and binding subtrees

## Summary

The repo still contains oversized legacy files, but the repo-wide governance mismatch this debt tracked is now retired. `tools/agent/structure-rules.yaml` and `.golangci.agent.yml` remain the executable ratchets for inherited hotspot debt, while every remaining maintained hotspot family now also has a nearest local `AGENTS.md` owner indexed from `docs/agent/local-rules.md`. Root harness docs no longer maintain a partial hardcoded directory list for those hotspots. Further extraction work now belongs to the local subtree owners or to a new focused debt entry, not to one umbrella placeholder.

## Evidence

- [AGENTS.md](../../../AGENTS.md)
- [docs/agent/architecture-guardrails.md](../architecture-guardrails.md)
- [docs/agent/module-boundaries.md](../module-boundaries.md)
- [docs/agent/local-rules.md](../local-rules.md)
- [tools/agent/repo-manifest.yaml](../../../tools/agent/repo-manifest.yaml)
- [tools/agent/task-matrix.yaml](../../../tools/agent/task-matrix.yaml)
- [tools/agent/skill-registry.yaml](../../../tools/agent/skill-registry.yaml)
- [tools/agent/structure-rules.yaml](../../../tools/agent/structure-rules.yaml)
- [tools/linter/go/.golangci.agent.yml](../../../tools/linter/go/.golangci.agent.yml)
- [tools/agent/scripts/agent_doc_validation.py](../../../tools/agent/scripts/agent_doc_validation.py)
- [src/semantic-router/pkg/dsl/AGENTS.md](../../../src/semantic-router/pkg/dsl/AGENTS.md)
- [src/semantic-router/pkg/apiserver/AGENTS.md](../../../src/semantic-router/pkg/apiserver/AGENTS.md)
- [src/semantic-router/pkg/cache/AGENTS.md](../../../src/semantic-router/pkg/cache/AGENTS.md)
- [src/semantic-router/pkg/memory/AGENTS.md](../../../src/semantic-router/pkg/memory/AGENTS.md)
- [src/semantic-router/pkg/looper/AGENTS.md](../../../src/semantic-router/pkg/looper/AGENTS.md)
- [src/semantic-router/pkg/modelselection/AGENTS.md](../../../src/semantic-router/pkg/modelselection/AGENTS.md)
- [src/semantic-router/pkg/selection/AGENTS.md](../../../src/semantic-router/pkg/selection/AGENTS.md)
- [src/semantic-router/pkg/tools/AGENTS.md](../../../src/semantic-router/pkg/tools/AGENTS.md)
- [src/semantic-router/pkg/responsestore/AGENTS.md](../../../src/semantic-router/pkg/responsestore/AGENTS.md)
- [src/semantic-router/pkg/imagegen/AGENTS.md](../../../src/semantic-router/pkg/imagegen/AGENTS.md)
- [src/semantic-router/pkg/promptcompression/AGENTS.md](../../../src/semantic-router/pkg/promptcompression/AGENTS.md)
- [deploy/addons/redis/AGENTS.md](../../../deploy/addons/redis/AGENTS.md)
- [src/training/model_eval/AGENTS.md](../../../src/training/model_eval/AGENTS.md)
- [src/training/model_classifier/AGENTS.md](../../../src/training/model_classifier/AGENTS.md)
- [candle-binding/src/core/AGENTS.md](../../../candle-binding/src/core/AGENTS.md)

## Why It Matters

- Large hotspot files are still debt, not precedent, but they no longer need one cross-cutting debt entry just to explain who owns the ratchet posture.
- Without nearest local rules, changed-file lint relief and structure-rule exceptions become hard to reason about, because contributors can see the executable carve-out without seeing the narrow subtree boundary that justifies it.
- A second partial hotspot inventory in root docs drifts quickly. Once that list diverges from the indexed local-rule inventory, contributors stop knowing which guidance is canonical.
- Closing this umbrella debt makes the source of truth sharper: executable ratchets live in the harness, local hotspot ownership lives in indexed subtree `AGENTS.md` files, and any future durable architecture gap must be tracked in a focused debt entry instead of a repo-wide placeholder.

## Desired End State

- Structural limits remain global, while each maintained hotspot family has a nearest local `AGENTS.md` supplement that explains the extraction-first ownership model for follow-up work.
- Root harness docs point hotspot work at the canonical local-rules index instead of maintaining a second partial directory inventory.
- Remaining oversized files continue to be governed by executable ratchets plus local subtree rules. If one hotspot family needs more than that, it gets its own focused debt entry.

## Exit Criteria

- Satisfied on 2026-04-06: `docs/agent/local-rules.md`, `tools/agent/repo-manifest.yaml`, `tools/agent/task-matrix.yaml`, and `tools/agent/skill-registry.yaml` now index nearest local `AGENTS.md` owners for the remaining hotspot families that still carry structural ratchets, including DSL, API server, cache, memory, looper, modelselection, selection, tools, responsestore, image generation, prompt compression, Redis addon wiring, training evaluation or classifier scripts, and Candle binding core support code.
- Satisfied on 2026-04-06: [AGENTS.md](../../../AGENTS.md) and [module-boundaries.md](../module-boundaries.md) now route hotspot work through the canonical [local-rules.md](../local-rules.md) index instead of maintaining a second partial directory list.
- Satisfied on 2026-04-06: the repo-wide ownership mismatch this debt tracked is gone. Remaining oversized files are still debt, but they are now governed by executable ratchets plus nearest local subtree rules, so further follow-up belongs in the owning subtree or a new focused debt entry rather than in this umbrella item.

## Retirement Notes

- Added local `AGENTS.md` owners for the remaining hotspot families that still depend on structural ratchets: `pkg/dsl`, `pkg/apiserver`, `pkg/cache`, `pkg/memory`, `pkg/looper`, `pkg/modelselection`, `pkg/selection`, `pkg/tools`, `pkg/responsestore`, `pkg/imagegen`, `pkg/promptcompression`, `deploy/addons/redis`, `src/training/model_eval`, `src/training/model_classifier`, and `candle-binding/src/core`.
- Updated the harness inventories and routing surfaces so those new local rules are discoverable, validated, and selected by `make agent-report` like the older hotspot supplements.
- Simplified root hotspot discovery by pointing both [AGENTS.md](../../../AGENTS.md) and [module-boundaries.md](../module-boundaries.md) at the indexed [local-rules.md](../local-rules.md) inventory instead of repeating a partial directory list.
- Remaining oversized files are intentionally still called debt in [architecture-guardrails.md](../architecture-guardrails.md), [structure-rules.yaml](../../../tools/agent/structure-rules.yaml), and [.golangci.agent.yml](../../../tools/linter/go/.golangci.agent.yml). This entry is closed because the repo-wide governance mismatch is retired, not because every local hotspot is already below the structural thresholds.

## Validation

- `make agent-validate`
- `make agent-ci-gate AGENT_CHANGED_FILES_PATH=/tmp/vsr_td006_changed.txt`
