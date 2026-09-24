---
title: Overview
---

# Fleet Sim overview

Fleet Sim answers planning questions that are expensive to explore on a live
GPU fleet: how many workers a workload may need, where to split traffic across
pools, and which assumptions most affect a latency or cost target.

It provides the `vllm-sr-sim` command-line tool and an optional standalone HTTP
service for automation or custom planning clients.

## What goes into a study

A useful study combines four kinds of input:

- a workload distribution, or an uploaded trace from which the service derives
  a token-length distribution;
- a service objective, currently reported primarily as P99 time to first token
  (TTFT);
- one or more pools with GPU profile, count, and maximum context length; and
- a routing policy, such as a prompt-length split or a model-to-pool mapping.

The simulator can then compare fleet size, queueing delay, utilization, modeled
cost, and, when a power profile is available, estimated energy behavior.

## Two levels of analysis

Fleet Sim uses analytical sizing for fast searches and a discrete-event
simulator (DES) for request-by-request checks.

1. Analytical sizing narrows a large configuration space quickly.
2. DES tests selected candidates with explicit arrivals, queueing, prefill,
   decode, and KV-cache admission.

Agreement between the two is useful evidence, but neither is a production
guarantee. Validate a final design with traces and measurements from the model,
hardware, tensor-parallel layout, vLLM configuration, and runtime version you
will deploy.

## Built-in data is a starting point

The repository includes workload CDFs and GPU profiles so you can learn the
workflow without collecting data first. Their constants are planning defaults,
not current prices or benchmark results for every model. Cost, latency, KV
capacity, and power conclusions become deployment-specific only after those
inputs are calibrated.

This distinction matters most for hardware comparisons: some built-in profiles
represent different model sizes and parallel layouts. Comparing their output
directly can measure the combined system choice, but it does not isolate the GPU
itself.

The HTTP service converts an uploaded trace to a total-token CDF and generates
Poisson arrivals for simulation. It does not replay the trace's original
inter-arrival times or per-request route labels. The Python library's
`TraceWorkload` is available when an exact timestamped replay is required.

## What Fleet Sim does not do

- It is not part of the router's live inference path.
- It does not autoscale or curtail a running deployment.
- It does not model kernels, networking, failures, or scheduler behavior with
  the fidelity of measurements from the target system.
- It does not decide whether a smaller model has acceptable answer quality.
- It does not replace load testing before capacity is committed.

## Choose a workflow

| Goal | Start here |
| --- | --- |
| Run a first sizing study | [Getting started](./getting-started) |
| Work through a planning decision | [Capacity-planning workflows](./use-cases) |
| Understand equations and simulator behavior | [Simulation model](./sim-algorithms) |
| Calibrate energy estimates | [Power model](./power-model) |

The [research context](./related-work) explains how this planning tool
relates to serving engines, high-fidelity simulators, and autoscaling systems.
