---
title: Connect the shared worker
---

# Connect the shared worker

sr-bench runs evaluations through one durable service that the CLI and
**Dashboard → Evaluation** share. This page covers the worker that
`vllm-sr serve` manages, a dedicated worker for the code and agent benchmarks,
and a standalone service for development.

`vllm-sr serve` starts an independent core worker alongside Dashboard. Its store
is `<state-root>/.sr-bench/<stack>/store` and its host API is loopback port
`8090 + stack port offset`. The CLI discovers that store and its private token
from the same workspace. Dashboard reloads and configuration replacement preserve
the worker. `vllm-sr stop` stops it without deleting saved evidence.

To avoid a host port conflict, set `VLLM_SR_BENCH_PORT` to an absolute port from
1 through 65535 for both `vllm-sr serve` and `vllm-sr benchmark`. This overrides
only the worker's loopback host port; its container port remains 8090 and the
stack port offset is not added to the override.

When only the selected Dashboard image changes, `serve` upgrades its managed
worker after verifying the same launch settings, store and credentials. The CLI
briefly pauses the worker to check its durable journal before replacing it.
Active runs and dataset preparations block the upgrade and resume unchanged.
Finish or cancel runs and wait for preparations to finish before retrying.
Saved results remain in the same store. A stopped worker, changed
credentials or changed launch settings still require explicit reconciliation.
The container runtime must support pausing for this image upgrade.

The core container has no Docker socket or GPU passthrough and does not include
all upstream harness dependencies. For the code and agent adapters, prepare a
dedicated host worker with pinned interpreters, source checkouts and sandbox
images. Inspect prerequisites with `vllm-sr benchmark setup --benchmark all`;
add `--install` to explicitly install the pinned optional environments and fetch
SciCode test data with a verified SHA256. Add `--build-sandbox` to build the
offline code-grading image; its receipt records the image and base-image digests
and pinned dependencies. Neither setup operation calls a model. The default
cache is `~/.cache/vllm-sr/sr-bench-1.0`, overridable with `SR_BENCH_HOME`.
SciCode data defaults to `assets/scicode/test_data.h5` inside that cache;
`SR_BENCH_SCICODE_TEST_DATA` can select another prepared file. Terminal task
images, source access, judges and simulators still need their declared
preparation. Configure `SR_BENCH_URL` to use that worker instead of creating the local
container. The URL must be reachable from each client; a host-local address and
a Dashboard-container address can differ while referring to the same service.

For standalone development:

```bash
# Provision SR_BENCH_TOKEN privately in both service and client environments.
vllm-sr benchmark --store ./data/sr-bench serve --host 127.0.0.1 --port 8090
```

In another terminal:

```bash
export SR_BENCH_URL=http://127.0.0.1:8090
vllm-sr benchmark --no-autostart runs
```

Non-loopback service binding requires `SR_BENCH_TOKEN`. Use authenticated,
private deployment wiring; browser users access the Dashboard proxy, not the
worker directly. `SR_BENCH_TOKEN_ENV` can name a custom service credential.
Model credentials use separate `api_key_env` references. Keep their values out
of manifests, command arguments and public artifacts.

## Next

- [Prepare reusable tasks and targets](./tasks-and-targets.md)
