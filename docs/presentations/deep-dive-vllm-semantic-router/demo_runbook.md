# Demo Runbook

## Goal

Reinforce the architectural story with one short demo path that proves the repo
connects runtime, dashboard, topology, and a maintained routing profile.

## Target Length

- ideal: `5` minutes
- hard cap: `8` minutes

## Demo Story

Show one product path:

1. local lifecycle is unified under `vllm-sr`
2. dashboard is the control-plane shell
3. topology and playground expose routing behavior
4. the AMD `balance` profile gives a concrete mental model for decisions

## Preflight

- confirm the local environment you plan to use:
  - `cpu-local` for the simplest talk path
  - `amd-local` only if you have the real ROCm environment available
- if doing a live stack startup, prefer the repo-native command:

```bash
vllm-sr serve --image-pull-policy never
```

- if the stack is already running, verify:

```bash
vllm-sr status
```

- have these tabs ready:
  - dashboard home
  - topology page
  - playground page
  - local copy of `deploy/recipes/balance.yaml`

## Demo Path

### Step 1 — Show the lifecycle entrypoint

Command:

```bash
vllm-sr status
```

Narration:

- the same CLI owns serve, logs, dashboard, status, and stop
- local runtime is a product surface, not a pile of scripts

### Step 2 — Open the dashboard

Target:

```text
http://localhost:8700
```

Narration:

- this is the operational shell
- config, topology, playground, and monitoring are all in one place

### Step 3 — Show topology

Narration:

- tie the visual graph back to the talk’s routing pipeline
- point out that this is why the architecture is explainable to humans

### Step 4 — Show playground

Narration:

- send one representative request
- explain that this route is going through the same control-plane stack, not a
  detached demo endpoint

### Step 5 — Open `balance.yaml`

Narration:

- show that the talk’s abstract routing tiers map to a maintained config asset
- point at a projection or decision that the audience already heard about

### Step 6 — Close the loop

Narration:

- restate the strongest point:
  the repo is interesting because runtime, config, dashboard, and deployment
  paths are aligned around one routing story

## Safe Demo Variants

### Variant A — Live runtime

Use when the local stack is already healthy and startup time is predictable.

### Variant B — Prestarted runtime

Preferred default for talks.

- start the stack before the session
- only show `status`, dashboard, topology, and playground during the talk

### Variant C — No live runtime

If the environment is unstable:

- keep the main deck
- show `balance.yaml`
- use appendix slides
- talk through the expected dashboard path instead of forcing a broken live run

## Failure Matrix

### Dashboard unavailable

- fall back to appendix slides and `balance.yaml`
- do not spend time debugging during the talk

### Playground broken

- stay on topology and config
- explain the route path conceptually

### Local stack startup slow

- use a prestarted stack for real talks
- avoid showing container boot logs live

### AMD environment unavailable

- keep AMD as a documented case study only
- do not promise a live ROCm run unless you have verified it beforehand

## Commands Cheat Sheet

```bash
vllm-sr serve --image-pull-policy never
vllm-sr status
vllm-sr logs router
vllm-sr dashboard
vllm-sr stop
```

## Repo Anchors

- `src/vllm-sr/README.md`
- `dashboard/README.md`
- `deploy/amd/README.md`
- `deploy/recipes/balance.yaml`
