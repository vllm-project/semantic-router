# Technical Debt Register

This document is the landing page for durable gaps between the repository's desired architecture and the current implementation. Use it to understand the workflow and find the canonical debt entry inventory.

## Why This Exists

- Some architectural gaps are too broad to fix in the same change that discovers them.
- If those gaps stay only in PR text, chat, or memory, agents and contributors will miss them.
- A durable debt register lets the harness distinguish:
  - canonical rules we want to converge toward
  - known implementation debt that has not been retired yet

## Canonical Files

- [tech-debt/README.md](tech-debt/README.md)
  - inventory, template, and entrypoint for editing tracked debt
- `docs/agent/tech-debt/*.md`
  - the only source of truth for per-item debt status, scope, summary, evidence, and exit criteria

## Policy

- When current code materially diverges from the desired architecture or harness rules and the gap is not fully closed in the same change, add or update the matching detailed entry under `docs/agent/tech-debt/`.
- Use stable IDs (`TD001`, `TD002`, ...) so PRs and follow-up work can point to the same debt item.
- Do not duplicate per-item status, scope, or summary in this landing page.
- Do not use this file for one-off branch tasks or temporary debugging notes.

## How to Retire Debt

- Close an item only when the underlying architectural gap is materially reduced, not just renamed.
- When a debt item is retired:
  - update the relevant canonical docs and executable rules first
  - update the matching debt entry file and the inventory in `docs/agent/tech-debt/README.md`
  - mark the entry as closed or remove it from the inventory when appropriate
  - reference the retiring PR or change in the entry if useful
