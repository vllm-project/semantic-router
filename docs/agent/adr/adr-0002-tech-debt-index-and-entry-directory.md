# ADR 0002: Organize Technical Debt Around Per-Item Entries and a Lightweight Register

## Status

Accepted

## Context

`docs/agent/tech-debt-register.md` started as a single-file record for durable unresolved architecture and harness gaps. As the debt inventory grows, keeping every item's evidence, impact, and exit criteria in one file makes the document harder to scan, harder to merge cleanly, and harder to extend without turning the register into another monolithic handbook.

The harness already uses indexed directory patterns for ADRs and execution plans. Technical debt needs similar scalability. It also benefits from a stable landing page that explains the workflow without duplicating per-item metadata.

## Decision

Use a split debt model with one metadata source:

- `docs/agent/tech-debt-register.md` is a lightweight landing page for debt policy and navigation.
- `docs/agent/tech-debt/README.md` defines the template and current entry inventory.
- each debt item lives in its own `docs/agent/tech-debt/td-001-*.md` file.
- per-item debt metadata lives only in the entry files, not in the landing page.
- `make agent-validate` and `make agent-scorecard` enforce and consume this entry-centric model.

## Consequences

- The debt inventory scales better as items grow in count and detail.
- Individual debt items can evolve with smaller diffs and fewer merge conflicts.
- Contributors still get one stable landing page, but debt status and scope are managed in only one place.
- The harness has one more indexed directory to keep in sync, so docs and executable validation must evolve together.
