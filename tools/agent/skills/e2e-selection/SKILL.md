---
name: e2e-selection
category: fragment
description: Selects which local and CI end-to-end test profiles are affected by a code change, using the repo-local profile map. Use when a change could affect E2E test behavior and the correct test profiles need to be identified and executed.
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

## Workflow

1. Read the E2E profile map to understand which profiles cover which surfaces
2. Run `make agent-e2e-affected CHANGED_FILES="..."` to identify affected profiles
3. Run `make e2e-test E2E_PROFILE=<profile>` for each affected profile
4. Verify local and CI E2E expectations are explicit and match the profile map

## Must Read

- [tools/agent/e2e-profile-map.yaml](../../e2e-profile-map.yaml)

## Standard Commands

- `make agent-e2e-affected CHANGED_FILES="..."`
- `make e2e-test E2E_PROFILE=<profile>`

## Acceptance

- Local and CI E2E expectations are explicit and match the profile map
