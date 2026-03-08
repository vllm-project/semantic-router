# ADR 0002: Organize Technical Debt as a Summary Register Plus Per-Item Entries

## Status

Accepted

## Context

`docs/agent/tech-debt-register.md` started as a single-file record for durable unresolved architecture and harness gaps. As the debt inventory grows, keeping every item's evidence, impact, and exit criteria in one file makes the document harder to scan, harder to merge cleanly, and harder to extend without turning the register into another monolithic handbook.

The harness already uses indexed directory patterns for ADRs and execution plans. Technical debt needs similar scalability, but it still benefits from one top-level summary page that makes the open inventory easy to review.

## Decision

Use a split debt model:

- `docs/agent/tech-debt-register.md` remains the canonical summary index.
- `docs/agent/tech-debt/README.md` defines the template and current entry inventory.
- each debt item lives in its own `docs/agent/tech-debt/TDxxx-*.md` file.
- the register summary and the detailed entry file must be updated together.
- `make agent-validate` and `make agent-scorecard` enforce and consume this indexed model.

## Consequences

- The debt inventory scales better as items grow in count and detail.
- Individual debt items can evolve with smaller diffs and fewer merge conflicts.
- Contributors still get one summary register for quick scanning.
- The harness has one more indexed directory to keep in sync, so docs and executable validation must evolve together.
