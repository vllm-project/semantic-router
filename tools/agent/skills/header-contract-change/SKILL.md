---
name: header-contract-change
category: primary
description: Adds, renames, removes, or changes the meaning of `x-vsr-*` HTTP headers and updates the downstream reveal/display path in dashboard and playground surfaces. Use when modifying router header contracts, changing how routing metadata is emitted in headers, or updating UI header allowlists.
---

# Header Contract Change

## Trigger

- Add a new `x-vsr-*` header
- Rename, remove, or change the meaning of an existing router header
- Change how dashboard or playground surfaces reveal routing metadata

## Workflow

1. Read change surfaces and feature-complete checklist for header contract dependencies
2. Modify header constants, emission logic, or UI allowlists
3. Run `make agent-report ENV=cpu CHANGED_FILES="..."` to identify impacted surfaces
4. Run `make agent-ci-gate CHANGED_FILES="..."` to validate alignment
5. Verify header constants, emission, `chatRequestSupport`, `HeaderDisplay`, `HeaderReveal`, and any topology consumers remain aligned with relevant test coverage

## Gotchas

- Header rename or removal is a downstream contract change; update display allowlists, docs, and tests in the same change.
- New matched-signal headers often need topology and playground updates in the same patch, not just the header constant and emitter.
- Avoid exposing unstable internal metadata as a user-visible header just because it is easy to emit.

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/feature-complete-checklist.md](../../../../docs/agent/feature-complete-checklist.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`

## Acceptance

- Header constants, emission, reveal allowlists, and topology or other user-visible consumers remain aligned
- Relevant tests cover the new or changed header contract
