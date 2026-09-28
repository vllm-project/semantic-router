---
name: router-contract-change
description: Use when changing a request-visible router contract across configuration, signals, decisions, algorithms, or recipe-scoped plugins in vLLM Semantic Router.
---

# Router contract changes

Read the nearest `AGENTS.md` for every touched subtree and
`tools/agent/docs/change-surfaces.md` for the affected contract.

Keep the request-facing entrypoint, isolated recipe, signal/projection inputs,
decision or algorithm, and selected backend/plugin behavior consistent. Update
canonical config, generated or schema-derived artifacts, and user-visible docs
only where that contract changes.

Run `make check` first. Use `make verify DOMAIN=<domain>` for the affected
integration contract, or `make verify PROFILE=<profile>` when behavior crosses
the deployment boundary. A pure refactor does not need new E2E coverage.
