---
title: Research Context
---

# Research context

Fleet Sim is a fleet-planning tool. It borrows familiar abstractions from
queueing, inference simulation, heterogeneous serving, and disaggregated
prefill/decode research, but it does not reproduce any one research system.

This page helps choose the right level of tool. It deliberately avoids copying
paper benchmark numbers into product guidance; performance claims are tied to
each paper's workload and evaluation environment.

## Where Fleet Sim fits

| Layer | Primary question | Fleet Sim coverage |
| --- | --- | --- |
| Serving engine | How should one replica batch and schedule tokens? | Represented through a calibrated profile, not simulated at kernel fidelity |
| Replica configuration | Which tensor/pipeline parallel and runtime settings should one replica use? | Input assumption; `ComputedProfile` can explore rough sensitivity |
| Fleet planning | How many pool instances are needed and how should traffic be split? | Primary scope |
| Runtime control | When should a live fleet scale, spill traffic, or reduce load? | Can evaluate static scenarios; does not operate the controller |
| Facility energy | What is the whole-system power and grid impact? | GPU board-power estimate only |

Use a profiler or high-fidelity engine simulator to calibrate a replica, Fleet
Sim to compare fleet topologies, and a production load test to accept the
result.

## Adjacent research

### Heterogeneous fleet selection

[Mélange](https://arxiv.org/abs/2404.14527) studies cost-aware selection of GPU
types across workload slices. Its core question is which measured hardware
profiles to combine. Fleet Sim focuses on pool counts, routing, and queueing
once performance profiles and costs have been supplied.

The common lesson is that a GPU SKU cannot be ranked without workload size,
arrival rate, SLO, model, and price. Fleet Sim's built-in profile names should
therefore be treated as inputs to replace, not a universal hardware ranking.

### Per-replica and engine simulation

[Vidur](https://arxiv.org/abs/2405.05465) uses profiled operation-level models
to simulate an LLM serving engine and search its configuration. This is a
higher-fidelity layer than Fleet Sim's `W`/`H` request-level profile.

[AIConfigurator](https://arxiv.org/abs/2601.06288) explores model and engine
configuration using hardware- and operation-level performance information.
Fleet Sim's computed profile has a roofline-style decomposition, but that does
not make it a validated substitute for AIConfigurator or a kernel database.

Measurements or selected settings from this class of tool can be converted
into a `ManualProfile` before fleet sizing.

### Disaggregated prefill and decode

[DistServe](https://arxiv.org/abs/2401.09670) and
[Splitwise](https://arxiv.org/abs/2311.18677) study systems that separate the
prefill and decode phases. They model or measure details such as phase
interference, parallel configuration, placement, and KV transfer.

Fleet Sim's `disagg` command is much narrower: it applies fixed phase
degradation and TTFT correction factors while sweeping prefill and decode
worker counts. It is useful for a first sensitivity study, not for validating
network topology or KV-transfer latency.

### Autoscaling and burst control

[SageServe](https://arxiv.org/abs/2502.14617) considers forecast-aware runtime
capacity control, while [TokenScale](https://arxiv.org/abs/2512.03416) considers
runtime scaling for disaggregated inference using stage-level token demand.

Fleet Sim does not implement either control loop. Its `whatif` output can help
identify static rates and fleet shapes that a separate controller should test,
but it does not model startup time, forecast error, control delay, or live
backpressure.

### Power-aware inference

Power-aware serving work, including
[GPU-to-Grid](https://arxiv.org/abs/2602.05116), motivates studying the latency
effect of concurrency caps. Fleet Sim's `grid-flex` command represents this as
a fitted or estimated batch-versus-board-power curve plus analytical and
optional DES latency.

It does not measure facility power or implement demand response. See the
[power model](./power-model) for calibration and reporting boundaries.

## Queueing foundation

Analytical sizing uses Erlang-C waiting probability and a Kimura-style M/G/c
tail approximation. The DES then evaluates selected candidates with explicit
synthetic arrivals and KV-slot admission.

These models answer a different question from kernel or serving-engine
simulation: they estimate fleet queueing once a service-time distribution is
known. Their assumptions—stationary Poisson arrivals, approximate independent
service times, and logical KV slots—must be checked against the deployment.

## Selecting a tool

| If you need to decide... | Start with... |
| --- | --- |
| batching, scheduler, or parallel settings for one replica | a serving-engine benchmark or high-fidelity simulator |
| which measured GPU/model profile is cheapest | heterogeneous configuration search |
| short/long pool counts and routing sensitivity | Fleet Sim `optimize`, `pareto`, and `simulate` |
| prefill/decode worker-count sensitivity | Fleet Sim `disagg`, followed by a disaggregated-system test |
| second-by-second scaling behavior | a runtime autoscaling/controller model |
| board-power sensitivity to concurrency | Fleet Sim power commands after measurement |
| production capacity approval | a production-shaped load test |

No tool in this table removes the need to preserve its workload, model,
hardware, runtime, and measurement assumptions with the result.
