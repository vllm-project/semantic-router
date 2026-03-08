---
name: header-contract-change
category: primary
description: Use when adding or changing request or response header contracts and the downstream user-visible reveal or display path.
---

# Header Contract Change

## Trigger

- Add a new `x-vsr-*` header
- Rename, remove, or change the meaning of an existing router header
- Change how dashboard or playground surfaces reveal routing metadata

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/feature-complete-checklist.md](../../../../docs/agent/feature-complete-checklist.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`

## Acceptance

- Header constants, emission, and UI allowlists remain aligned
- Relevant tests cover the new or changed header contract
