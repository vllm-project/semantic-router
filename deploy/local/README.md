# HX local systemd deployment (CPU classifier on HX-2)

This directory holds the systemd assets for the HX-2 persistent CPU-mode
classifier deployment (`hx-classifier`, endpoint `:8100`). They sit beside
`envoy.yaml` as local deployment boundaries: host-level assets that are not
packaged Kubernetes/Helm manifests.

## Assets

- `hx-vllm-srun-cpu.service` — the systemd unit. Runs
  `vllm-srun serve ... --device cpu` for
  `Decision-2.0-Kai-0.6B@cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764` as
  `hx-classifier` on `0.0.0.0:8100`, co-resident with the qwen-x GPU
  workload (which keeps the GPU; the classifier uses none of it).
- `hx-vllm-srun-cpu.env` — its environment file. Deliberately contains no
  GPU allocator variables.
- `hx-port-free-check` — the unit's `ExecStartPre` guard: refuse to start if
  the port is already bound. Read-only; it never kills processes.

## Qualification basis

CPU mode was qualified on HX-2 (2026-11-24) against the GPU-era reference
answers: 105/105 identical choices (5 sequential repetitions × 21 fixtures),
max probability drift 0.0061, p50 ≈ 259 ms / p95 ≈ 263 ms per decision
request, concurrency-2 ≈ 398 ms wall, peak RSS ≈ 4.3 GiB, zero GPU memory
used. All figures are far inside the consumer budgets (3.5 s signal timeout,
4 s adapter cap, 10 s advice budget).

## Install (root on the target host)

```bash
systemd-analyze verify hx-vllm-srun-cpu.service   # before installing
install -m 0644 hx-vllm-srun-cpu.env /srv/hx-semantic-router/units/
install -m 0755 hx-port-free-check /srv/hx-semantic-router/bin/
install -m 0644 hx-vllm-srun-cpu.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now hx-vllm-srun-cpu.service
```

The retired GPU unit (`hx-vllm-srun.service`) and its residency watchdog
(`hx-vllm-srun-watchdog.service`) must be stopped and disabled before this
unit binds port 8100; the port-free guard enforces that at start time.
