# Deployment and model-pool loop

## Install and inspect

Install a released CLI or build the repository's CLI through its documented
development target. Confirm the runtime platform, container engine, available
accelerators, ports, storage, and credential environment before serving.

Read an active config before editing it; use `vllm-sr config init` only for a
fresh workspace. Discover exact fields from the running schema rather than
from memory. Define each provider, model, backend reference, Model Card,
Entrypoint, Recipe, and listener explicitly. A model joins a route only when a
decision references it through `modelRefs`. Custom physical models and virtual
models use the same evaluation-record and routing-index contracts.

## Serve

Use the supported platform flag for the host, for example:

```bash
vllm-sr serve --platform amd --config config.yaml
```

Do not declare the deployment ready merely because the process started. Wait
for `/ready`, inspect `/startup-status`, list `/api/v1/inventory/models`, and
send a direct request to every physical backend before testing routing.

## Add or replace models

1. Establish backend health and protocol compatibility.
2. Measure serving latency, throughput, token usage, failure rate, and cost.
3. Add the physical model and Model Card to candidate YAML.
4. Run config validate and plan.
5. Apply, then route-preview representative cases.
6. Probe each direct model alias and every affected virtual model.
7. Run the required full benchmarks.
8. Optimize the Recipe only from comparable evidence.

Never estimate a virtual model's quality from member scores. Evaluate the
virtual model endpoint over the same suite so routing failures, retries, model
mix, latency, and cost remain observable.
