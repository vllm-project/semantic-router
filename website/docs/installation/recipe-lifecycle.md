---
title: Managed Recipe Lifecycle
description: Plan, apply, and delete recipes with the vllm-sr CLI against the Router management API.
---

# Managed Recipe Lifecycle

The `vllm-sr recipe` commands are the CLI's stateful write path for the canonical Router configuration: they read a recipe file, validate it, and compare-and-swap it into the live configuration. What a recipe is, and how recipes relate to entrypoints and models, is covered in [Recipes](../tutorials/global/recipes). How the CLI fits the other configuration interfaces is covered in [Configuration Workflows](./configuration-workflows). This page covers the command lifecycle.

All recipe commands talk to the Router management API and share the same connection options:

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

`list` prints every managed recipe together with the collection ETag — the version handle that the write commands below use as their precondition. `get` reads one recipe and returns it with its own ETag.

## Validate

```bash
vllm-sr recipe validate recipe.yaml
```

Validation checks the file against the recipe contract without touching the live configuration. It is the same validation `plan` and `apply` run first.

## Plan

```bash
vllm-sr recipe plan recipe.yaml
```

`plan` validates the recipe and binds the plan to the collection ETag that is current at that moment. It writes nothing: the output is what `apply` would execute, together with the ETag the apply will require. Use it as the review step before a write.

## Apply

```bash
vllm-sr recipe apply recipe.yaml
```

`apply` validates the recipe, then compare-and-swaps it into the active configuration using the collection ETag read at the start of the command. If the configuration changed on the server in the meantime, the ETag precondition fails and nothing is written. Re-run `plan` against the new state and apply again if the change is still what you want.

## Delete

```bash
vllm-sr recipe delete my-recipe
```

`delete` compare-and-swaps one recipe out of the configuration, with the same ETag precondition as `apply`. Only an unreferenced recipe can be deleted: while any entrypoint or other configuration element still points at it, deletion is refused. Delete removes that recipe and nothing else — there is no cascade and no undo, so treat `plan` as the review step.

## Interaction with the Dashboard

The recipe commands and the Dashboard's configuration editors act on the same canonical configuration through the same management API. Keep one interface as the source of truth for a deployment and use the others to inspect rather than to overwrite independently. The ETag preconditions reject writes that raced with another interface, but they cannot resolve a disagreement about which interface is authoritative.
