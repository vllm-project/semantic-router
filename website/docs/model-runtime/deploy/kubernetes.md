---
title: Run on Kubernetes
sidebar_label: Run on Kubernetes
sidebar_position: 3
description: Run the model runtime inside the Router pod on CPU or GPU, size its model volume, and share one GPU runtime between several Routers.
---

# Run on Kubernetes

The Router image already contains the CPU runtime, so managed deployments work
in any cluster. For GPUs, use the ROCm or CUDA Router image and give the Router
pod a GPU; to share GPU models between Routers, run the runtime as its own
Deployment and attach every Router to it.

```bash title="Write the GPU image and limit into the chart's values"
vllm-sr serve --target kubernetes --platform rocm
```

## Models on the Router pod's GPU

The ROCm Router image, `ghcr.io/vllm-project/semantic-router/vllm-sr-rocm`,
contains the runtime with PyTorch for ROCm, and `vllm-sr-cuda` the runtime with
PyTorch for CUDA:

1. Give the Router pod a GPU (`amd.com/gpu` or `nvidia.com/gpu` in its
   resource limits).
2. Set `device: rocm:0` or `device: cuda:0` on a deployment, and the Router
   runs that model on the GPU itself.

`vllm-sr serve --target kubernetes --platform rocm|cuda` writes the image and
the GPU limit into the chart's values for you.

## The model volume

On first start the runtime downloads the models a Router uses into its model
volume (`/app/models`, the chart's `persistence` claim, 10 GiB by default).

- Size the claim for the models you route with: Vela Omni Mini alone takes
  4.3 GB.
- In an air-gapped cluster, fetch the models into it first
  ([Troubleshooting](./troubleshooting.md#the-runtime-stays-in-loading-or-warming)).

## CPU models in the ROCm image

:::warning
A Router whose models all run on the CPU should use the CPU image.
:::

- Models you keep on the CPU in the ROCm image run on its ROCm build of
  PyTorch. For most Vela task models that build fails the runtime's load-time
  check that a model answers the same alone and in a batch, so they batch as
  GPU models do: concurrent requests share forwards after the batching window,
  and an answer can move by rounding.
- The ROCm image's PyTorch has no CPU LAPACK, so decoders with gated delta rule
  layers (the Qwen3.5-based models, such as Decision 2.0 Lux-9B or Vela 2.0 4B)
  can't load on its CPU. The runtime refuses `device: cpu` for them at startup,
  before their weights load, and names the cause. Serve them on the GPU, or
  from the CPU image.

## Share a runtime between Routers

To share GPU models between several Routers, run the runtime as its own
Deployment from the same image and attach every Router to its Service. The
image's `vllm-srun` command starts it:

```yaml title="model-runtime.yaml"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-runtime
spec:
  replicas: 1
  selector:
    matchLabels: {app: model-runtime}
  template:
    metadata:
      labels: {app: model-runtime}
    spec:
      containers:
        - name: runtime
          image: ghcr.io/vllm-project/semantic-router/vllm-sr-rocm:latest
          command: ["vllm-srun"]
          args: ["serve", "vllm-sr/Decision-2.0-Lux-9B", "--device", "rocm:0", "--host", "0.0.0.0", "--port", "8100"]
          resources:
            limits: {amd.com/gpu: 1}
          ports:
            - {name: http, containerPort: 8100}
          readinessProbe:
            httpGet: {path: /health, port: http}
          livenessProbe:
            httpGet: {path: /health/live, port: http}
---
apiVersion: v1
kind: Service
metadata:
  name: model-runtime
spec:
  selector: {app: model-runtime}
  ports:
    - {name: http, port: 8100, targetPort: http}
```

Set the deployment's `endpoint` to
`http://model-runtime.<namespace>.svc.cluster.local:8100`, as in
[Attach to a runtime you run](./connect-the-router.md#attach-to-a-runtime-you-run).

:::tip
Use `/health` as the readiness probe: it succeeds only after every model has
loaded and passed its self-check. `/health/live` answers while the process
serves requests, whether or not its models are ready.
:::
