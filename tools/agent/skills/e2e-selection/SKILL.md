---
name: e2e-selection
category: fragment
description: Affected local and CI E2E profile selection using the repo-local profile map.
---

# E2E Selection

## Trigger

- The primary skill changes behavior that could affect one or more E2E profiles

## Required Surfaces

- `local_e2e`
- `ci_e2e`

## Conditional Surfaces

- `local_smoke`

## Stop Conditions

- The affected profile cannot be determined from the current mapping and needs manual classification

## Must Read

- [tools/agent/e2e-profile-map.yaml](../../e2e-profile-map.yaml)

## Standard Commands

- `make agent-e2e-affected CHANGED_FILES="..."`
- `make e2e-test E2E_PROFILE=<profile>`

## Acceptance

- Local and CI E2E expectations are explicit and match the profile map
