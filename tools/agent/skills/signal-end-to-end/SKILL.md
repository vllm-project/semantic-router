---
name: signal-end-to-end
category: primary
description: Implements end-to-end signal changes spanning router config, signal extraction, CLI schema, optional bindings, dashboard surfaces, and E2E test coverage. Use when adding a new signal type, changing signal configuration or extraction logic, updating CLI schema for signal parameters, or modifying how signals are displayed in the dashboard.
---

# Signal End to End

## Trigger

- Add a new signal type or signal rule shape
- Change how an existing signal is configured, extracted, emitted, or displayed
- Touch router signal config plus Python CLI schema in the same feature

## Workflow

1. Read change surfaces, module boundaries, and playbooks for signal context
2. Modify signal config, extraction, CLI schema, and dashboard surfaces as needed
3. Run `make agent-report ENV=cpu CHANGED_FILES="..."` to identify all impacted surfaces
4. Run `make agent-ci-gate CHANGED_FILES="..."` to validate signal contract alignment
5. When the signal becomes user-visible, verify the visualization chain stays aligned across classify or eval payloads, emitted headers or looper headers, topology backend conversion, topology frontend parsers or constants, and playground reveal surfaces
6. Verify router config, signal extraction, and CLI schema stay aligned with relevant E2E coverage

## Gotchas

- Signal changes rarely stop at extraction; CLI schema, display surfaces, or bindings often need updates too.
- New matched-signal families often need more than one UI touchpoint: `matched_signals` JSON, `x-vsr-matched-*` headers, `chatRequestSupport`, `HeaderDisplay`, `HeaderReveal`, topology backend conversion, and topology frontend parser or constants.
- New signal semantics need explicit E2E ownership, not just unit-level classifier coverage.

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/module-boundaries.md](../../../../docs/agent/module-boundaries.md)
- [docs/agent/playbooks/go-router.md](../../../../docs/agent/playbooks/go-router.md)
- [docs/agent/playbooks/vllm-sr-cli-docker.md](../../../../docs/agent/playbooks/vllm-sr-cli-docker.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu CHANGED_FILES="..."`

## Acceptance

- Router config, signal extraction, and Python CLI schema stay aligned for the signal contract
- Any user-visible signal metadata updates the relevant classify or eval payloads, headers, topology surfaces, and playground reveal surfaces together
- Relevant E2E coverage is added or updated when behavior changes
