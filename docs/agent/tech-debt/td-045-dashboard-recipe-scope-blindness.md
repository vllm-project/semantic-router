# TD045: Dashboard Surfaces Are Blind to Named Recipes

## Status

Open.

## Owner Plan

[PL0038 Entrypoints and Recipes](../plans/pl-0038-entrypoints-recipes.md), task
T12 (dashboard surfaces).

## Release Relevance

Named recipes are the unreleased #2612/#2613 surface. Before they ship, every
dashboard read of routing decisions must either model recipe scope or state
that it intentionally shows the default profile only; today the gaps are
silent.

## Scope

- `dashboard/backend/handlers/topology_response.go`
- `dashboard/backend/handlers/setup.go`
- `dashboard/backend/configprojection/builder.go`

## Summary

The dashboard backend reads only the default routing profile. The router-side
read-site correction (#2724) fixed the router's whole-surface reads and pinned
the remaining default-profile reads in an allowlist, but the dashboard module
was out of that PR's scope: its decision, plugin, and signal views omit named
recipes entirely, and the setup wizard can reject a valid recipes-only config.

## Evidence

- `topology_response.go` (`appendEvaluatedRulesFromConfig`): the test-query
  panel replays only `DefaultDecisions`, so a request served by a named recipe
  shows evaluated rules from a profile it did not route through.
- `setup.go` (`summarizeSetupConfig` / `CanActivate`): the decision count comes
  from the exported top-level routing block, so a recipes-only config reads as
  "0 decisions" and the wizard refuses to activate it, while the model summary
  spans every recipe — one panel, two scopes.
- `configprojection/builder.go` (`BuildSnapshot`): only
  `canonical.Routing.Decisions`/`Signals`/`Projections` reach the snapshot;
  `canonical.Recipes` is dropped wholesale, so recipe decisions and their
  plugins never appear in the projection views.
- These sites do not spell `DefaultDecisions`, so the read-site allowlist
  guard (`src/semantic-router/pkg/config/default_decisions_allowlist_test.go`)
  structurally cannot flag them; only `topology_response.go` is registered
  there today.

## Why It Matters

Operators use the dashboard to verify what the router will do. A recipe
decision that routes, enables plugins, and reports in `/metrics` while staying
invisible in the dashboard reproduces the exact silent-divergence class that
issue #2723 removed from the router.

## Desired End State

Dashboard views either render recipe scope (grouped by profile, with
entrypoint context) or carry an explicit "default profile only" label; the
setup wizard activates recipes-only configs.

## Exit Criteria

- `BuildSnapshot` projects recipe decisions, plugins, and signals.
- `CanActivate` accepts a config whose decisions all live in named recipes.
- The topology test-query panel either models recipe scope or labels its
  default-profile limitation in the UI.
