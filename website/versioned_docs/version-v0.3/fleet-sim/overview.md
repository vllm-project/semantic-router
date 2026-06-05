---
title: Overview
---

# Fleet Sim Overview

Fleet Sim is the maintained fleet simulator for vLLM Semantic Router. The
`vllm-sr-sim` package is its CLI and service entrypoint. It helps you plan GPU
fleets before deployment, compare routing and split strategies, and expose those
workflows inside the dashboard without reviving a separate simulator frontend.

## What Fleet Sim is for

- sizing homogeneous, heterogeneous, or disaggregated fleets against a latency target
- comparing annualized cost across GPU choices, routing policies, and threshold choices
- validating planning assumptions with simulation runs, trace replay, and what-if analysis
- surfacing those workflows in the dashboard through a maintained backend proxy

## What Fleet Sim is not for

- it is not the router's live request path
- it is not a runtime autoscaler or burst controller
- it is not a per-kernel profiler for one deployment replica
- it is not a replacement for the router configuration docs

## Deployment modes

`vllm-sr-sim` can run as:

- a standalone Python CLI for local sizing and what-if analysis
- an HTTP service with `vllm-sr-sim serve`
- a sidecar container that `vllm-sr serve` starts by default on the shared `vllm-sr-network`

## Read this section in order

1. [Getting started](./getting-started.md) for local sidecar, standalone CLI, and external service setup
2. [Dashboard integration](./dashboard-integration.md) for the proxy path and UI surfaces
3. [Capacity planning scenarios](./use-cases.md) for example-driven decision workflows
4. [Simulation model reference](./sim-algorithms.md) and [power model reference](./power-model.md) when you need the underlying mechanics
5. [Guide PDF](pathname:///files/fleet-sim/fleet-sim.pdf) and [guide assets](./guide.md) when you want the printable version or source files
