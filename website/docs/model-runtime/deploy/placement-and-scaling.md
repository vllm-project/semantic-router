---
title: Place and scale workers
sidebar_label: Place and scale workers
sidebar_position: 2
description: Run independent replicas of a deployment on one or more GPUs, understand how requests are dispatched, and size CPU threads.
---

# Place and scale workers

A logical deployment can run as one or more independent **replicas**, each its
own worker process with its own copy of the weights. Use replicas to add
compute capacity on more GPUs; use CPU threads to size a worker on the CPU.

```bash title="Two workers of the default deployment on two GPUs"
vllm-sr serve vllm-sr/d3-edge -e --platform rocm -dp 2 --device-ids 0,1
```

![Per-deployment replica dispatch and the interfaces inside a model worker](/img/architecture/system-one/03-serving-engine-and-workers.svg)

## Place and scale replicas

`-dp` sets the default deployment's independent worker count. Its engine
arguments are separate from placement:

```bash
# Two workers on one explicitly selected host GPU.
vllm-sr serve vllm-sr/d3-edge -e --platform rocm -dp 2 --device-ids 0
# Two workers across two host GPUs.
vllm-sr serve vllm-sr/d3-edge -e --platform rocm -dp 2 --device-ids 0,1
```

- Without device IDs, a new GPU placement uses the first N available GPUs and
  refuses insufficient capacity; it does not silently place every worker on one
  card. Existing authored placements resize in their declared order.
- Specifying only MODEL preserves configured replicas and engine arguments.
- These flags write the same canonical resource shown below; there is no
  separate DP state.
- For Kubernetes, use allocation ordinals in config instead of host IDs.

```yaml title="config.yaml"
global:
  router:
    enabled: false
  model_catalog:
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/d3-edge
        replicas:
          - device: rocm:0
          - device: rocm:1
    system:
      decision_model:
        deployment: primary
```

:::warning
Replicas do not shard weights and are not tensor or pipeline parallelism. Use
different GPUs to add compute capacity: repeating a device explicitly starts
multiple workers on that GPU, and each worker needs memory for its model.
Within a worker, rows of concurrent requests already share forwards
([Engine arguments](./engine-args.md)). Measure throughput and tail latency
with your input lengths and concurrency before choosing a layout.
:::

### What replicas share

- All replicas of a deployment use the same artifact, revision, engine
  arguments and capabilities, except batch limits the deployment leaves unset:
  each GPU worker calibrates its own token budget at startup, so replicas may
  report different `max_num_batched_tokens`.
- Set `max_num_seqs` and `max_num_batched_tokens` to give every replica the
  same limits; a replica that reports other values than set is kept out of
  dispatch.
- Do not combine `replicas` with deployment-level `device` or `endpoint`; an
  attached replica names its served model with its own `model`.
- Omitting `replicas` keeps the one-worker shorthand, with the same dispatch
  and status contract as an explicit single replica.
- Single-worker deployments retain the frontend result cache; multiple
  replicas use independent worker caches.
- A replica's placement is independent of bindings, so changes to the signals
  that use a deployment retain unchanged worker identities. Reconfiguration
  prepares a compatible pool before publishing it, and retired workers drain
  existing requests.

### How requests are dispatched

A request's fused question batch is dispatched as one unit after the logical
model is selected. The pool selects the ready worker with the fewest
outstanding request bytes, breaking ties by the least recently assigned worker.
This is a local load estimate, not the attached runtime's queue length or a
measured token count. Native requests and routing tasks share the physical
worker's load accounting.

- The pool admits at most 32 in-flight HTTP requests per worker and has no pool
  waiting queue. It returns overload when all ready replicas are full.
- This limit is not GPU-forward concurrency: a forward already running may
  continue after its caller times out.
- Inventory reports desired and ready replica counts, per-worker readiness,
  in-flight requests and restarts. A degraded pool can keep serving through its
  healthy replicas.

### Eject a replica that keeps timing out

A worker whose forward hangs can keep answering `/health`, so readiness alone
does not take it out of dispatch. The pool counts, per replica, the exchanges
in a row that run out of their deadline, the caller's or the deployment's
`timeout_ms`, and ejects a replica that reaches `consecutive_timeouts`. Each
field is optional:

```yaml
global:
  model_catalog:
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-4B
        replicas:
          - device: rocm:0
          - device: rocm:1
        outlier_detection:
          consecutive_timeouts: 3     # default 3; 0 turns ejection off
          base_ejection_ms: 30000     # default 30000
          max_ejection_ms: 300000     # default ten times the base, at most an hour
          max_ejection_percent: 50    # default 50
```

- A cancellation, a refused request or a failure neither counts nor resets the
  count; a successful answer starts it again. An exchange past `timeout_ms`
  counts, and like a failure it also takes the replica out of dispatch for a
  short backoff (0.1 s, doubling to 3.2 s) and probes its `/health`, so a hung
  replica stays mostly out of dispatch even while ejection is off or held
  back.
- An ejected replica leaves dispatch for `base_ejection_ms`, and the Router
  probes its `/health` at once. When the cool-down ends the replica returns; its
  next timeout ejects it again for twice as long, up to `max_ejection_ms`. A
  successful answer readmits it at once and starts over.
- Ejection never takes the replicas that serve. A replica is ejected only while
  another ready replica has answered since its timeouts began, so a model that
  is slow on every replica keeps them all. At most `max_ejection_percent` of the
  replicas are out at once (one may always be), and a deployment with one worker
  never ejects it. If no other replica can take requests, the ejected ones serve
  again.
- Inventory reports an ejected replica with the state `ejected`. The Router logs
  `replica_ejected`, `replica_readmitted` and, when a rule above holds an
  ejection back, `replica_ejection_withheld`, and counts ejections and
  readmissions in its [metrics](../reference/metrics.md).

## CPU threads

Each managed CPU worker defaults to half the Router's available CPU budget,
rounded down, with at least one thread and at most 16. To choose a different
thread count per worker, set a positive integer before starting the stack:

```bash
VLLM_SRUN_CPU_THREADS=8 vllm-sr serve --platform cpu --config config.yaml
```

- The setting is capped by the available CPU budget; it does not set the number
  of workers. Each deployment or replica still owns an independent worker, and
  mode changes do not redistribute threads among active workers.
- Start with the default, then compare latency and throughput with your actual
  input lengths and concurrency. Fewer threads can reduce contention when
  several CPU models are busy; longer inputs on dedicated cores may benefit from
  a higher override.
- This is a per-worker limit, so several busy workers can still contend for the
  same cores. Reserve cores for other services when sharing a host.
- Runtimes you attach through `endpoint` manage their own threads.

### Learn the thread count from traffic

A managed CPU runtime can also learn the intra-op thread count its traffic
wants: the fastest count for a short input is often lower than for a long one,
and on a shared host it moves with the neighbors' load.

| Setting | What the worker does |
| --- | --- |
| `VLLM_SRUN_CPU_ADAPTIVE_THREADS=1` | Learns and reports. It serves every request at the configured count while the allowed counts run as timed shadow passes that must reproduce the served answer bit for bit, then logs the fastest count per input-size range. Point `--threads` at the learned optimum to apply it. |
| `VLLM_SRUN_CPU_ADAPTIVE_THREADS=2` | Also serves the learned counts, after they shadow-verify against the served answers for a transition window. |

<details>
<summary>How learning stays exact</summary>

- Exploration starts only after the startup golden check has passed. A warm
  pass at the configured count keeps the caches out of the comparison, and one
  count that drifts on this host retires the learning for good.
- With `1`, the worker's answers never leave the configured count. A single
  count is safe on every host, because only a change of count can move an
  answer.
- Whether answers survive a change of count is a property of the host and its
  kernels: bit-identical on the measured Arm hosts, input-dependent drift on
  x86. `2` is meant for hosts whose bit-identity across counts a record vouches
  for, because a host that drifts only on inputs the window never saw could
  still move an exact answer.
- All forwards state their thread count on a shared per-process context, so
  several workers in one process never inherit each other's count.
- The exploration asks something of the first requests, and the learned setting
  reflects the load the worker saw while learning; on a host whose load shifts,
  relearn by restarting the worker.

</details>
