---
title: Managed Recipe Lifecycle
description: Plan, apply, and delete recipes with the vllm-sr CLI against the Router management API.
---

# Managed Recipe Lifecycle

The `vllm-sr recipe` commands are the CLI's stateful write path for the
canonical Router configuration: they read a recipe file, validate it, and
compare-and-swap it into the live configuration. This page covers the command
lifecycle.

- **What a recipe is,** and how recipes relate to entrypoints and models:
  [Recipes](../tutorials/global/recipes).
- **How the CLI fits** the other configuration interfaces:
  [Configuration Workflows](./configuration-workflows).

The recipe commands that operate on the live configuration talk to the Router management API and share the same connection options:

| Option | Default | Meaning |
|---|---|---|
| `--endpoint` | local Router API port | Router management base URL |
| `--token-env` | `VSR_MGMT_TOKEN` | Environment variable holding the management token |
| `--timeout` | `15` | Request timeout, in seconds |

The token is read from the environment variable named by `--token-env`; the CLI never accepts a token as a command argument.

## Inspect

```bash
vllm-sr recipe list
vllm-sr recipe get my-recipe
```

`list` prints every managed recipe together with the collection ETag, which identifies the state observed by that read. `get` reads one recipe and returns it under the same collection ETag, because the ETag identifies the configuration document rather than any single recipe. The write commands below independently fetch a fresh collection ETag when they run.

## Validate

```bash
vllm-sr recipe validate recipe.yaml
```

Validation checks the file against the recipe contract without touching the live configuration. It is the same validation `plan` and `apply` run first.

## Plan

```bash
vllm-sr recipe plan recipe.yaml
```

`plan` validates the recipe and reports the proposed action, with the
collection ETag observed at that moment. It writes nothing.

- **Its ETag is advisory.** A later `apply` fetches a fresh ETag rather than
  requiring the one `plan` printed, so planning is not a lock, and doesn't
  guarantee the configuration stays unchanged.
- **Use it to review** the proposed apply. If the current configuration
  matters to the decision, run it again right before applying.

## Apply

```bash
vllm-sr recipe apply recipe.yaml
```

`apply` validates the recipe, then compare-and-swaps it into the active
configuration, using the collection ETag it read at the start of that command:

- **A race fails safely.** If the configuration changes between that read and
  the write, the ETag precondition fails and nothing is written.
- **An earlier `plan` doesn't hold the state.** A change after `plan` but
  before `apply` begins doesn't block the apply; it can succeed against the
  newer state. Re-run `plan` against the new state if the change needs review
  again.

## Delete

```bash
vllm-sr recipe delete my-recipe
```

Before deleting, inspect the target with `get` or `list`:
`recipe plan recipe.yaml` previews only an apply, not what `delete` removes.

- **A race fails safely.** `delete` fetches a fresh collection ETag and
  compare-and-swaps one recipe out of the configuration, so it protects only
  against a change that races with the delete command itself.
- **Only an unreferenced recipe.** While any entrypoint still points at a
  recipe, deletion is refused; an entrypoint mapping is the only configuration
  element that can reference a recipe.
- **Not the default recipe.** It's the top-level routing profile, and the
  server refuses it the same way.
- **No cascade, no undo.** Delete removes that recipe and nothing else.

## Interaction with the Dashboard

The recipe commands and the Dashboard's configuration editors act on the same
canonical configuration, through different paths:

| Path | How it writes |
| --- | --- |
| `vllm-sr recipe` | Through the Router management API; each write compare-and-swaps under a freshly fetched collection ETag. |
| Dashboard | Writes the configuration and propagates it to the runtime itself, outside the ETag precondition. |

Keep one interface as the source of truth for a deployment, and use the other
to inspect rather than to overwrite independently.
