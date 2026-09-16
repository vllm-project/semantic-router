# AMD ROCm deployment details

Read this only when deploying on AMD hardware. Follow the general
[deployment workflow](deployment-loop.md) for stack isolation, model sizing,
network access, and verification.

For local Docker, select `--platform amd` through the installed `serve` contract.
`VLLM_SR_AMD_ROUTER_VISIBLE_DEVICES` restricts the Router's visible GPUs; account
for generation backends separately and preserve the same selection on restart.
Verify host/container ROCm compatibility before loading models.

Inspect the effective model bindings and their referenced deployments using the
discovered schema. Confirm each required model's backend and device in startup
evidence and the live inventory; the AMD platform flag alone does not establish
GPU execution. Report unexpected CPU execution or a failed binding as an unmet
acceleration requirement.

Use the [AMD installation guide](https://vllm-sr.ai/docs/installation/amd-rocm)
for supported runtime options. If a model or kernel fails, retain its exact
image, model revision, error, and triggering input size. Verify a proposed fix
on the affected path before claiming success; short requests do not qualify a
long-context or concurrent workload. See [evaluation details](evaluation-loop.md)
when that qualification is part of the user's task.
