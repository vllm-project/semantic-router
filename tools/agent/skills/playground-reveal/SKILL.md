---
name: playground-reveal
category: fragment
description: Modifies playground chat rendering, reveal overlays, and user-visible route metadata display in the dashboard. Use when changing how routing decisions are presented to users, updating reveal overlay content, or adjusting playground response formatting.
---

# Playground Reveal

## Trigger

- The primary skill touches playground chat rendering, reveal overlays, or user-visible route metadata

## Workflow

1. Read change surfaces doc to understand playground reveal dependencies
2. Modify playground rendering, reveal overlays, or route metadata display
3. Run `make dashboard-check` to validate UI consistency
4. Verify reveal surfaces stay aligned with emitted metadata and user-visible expectations

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)

## Standard Commands

- `make dashboard-check`

## Acceptance

- Playground reveal surfaces stay aligned with emitted metadata and user-visible expectations
