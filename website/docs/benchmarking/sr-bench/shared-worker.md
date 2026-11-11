---
title: Connect the shared worker
---

# Connect the shared worker

sr-bench runs evaluations through one durable service that the CLI and
**Dashboard → Evaluation** share. This page covers the worker that
`vllm-sr serve` manages, a dedicated worker for the code and agent benchmarks,
and a standalone service for development.

## The worker `vllm-sr serve` manages

`vllm-sr serve` starts an independent core worker alongside Dashboard:

- **Store:** `<state-root>/.sr-bench/<stack>/store`. The CLI discovers it, and
  its private token, from the same workspace.
- **Host API:** loopback port `8090 + stack port offset`. To avoid a host port
  conflict, set `VLLM_SR_BENCH_PORT` to an absolute port from 1 through 65535
  for both `vllm-sr serve` and `vllm-sr benchmark`. That overrides only the
  worker's loopback host port: its container port stays 8090, and the stack
  port offset isn't added to the override.
- **Lifetime.** Dashboard reloads and configuration replacement keep the
  worker; `vllm-sr stop` stops it without deleting saved evidence.

`serve` reuses a running worker whose image, launch settings, store and
credentials are unchanged. When any of them changes, for example a new
Dashboard image, host alias, port or credential, `serve` replaces the worker
it started:

- **Checked first.** `serve` briefly pauses the worker to check the worker's
  own durable journal, so the container runtime must support pausing. Saved
  results stay in the store.
- **Left alone when busy.** If the journal shows active runs or dataset
  preparations, or can't be read, the worker resumes unchanged. A stopped
  worker, and a container without the sr-bench label, are also left as they
  are.
- **The stack still starts.** In each of those cases `serve` logs a warning
  and still starts the Router and Dashboard; run `serve` again once the worker
  is idle.

## A dedicated worker for code and agent benchmarks

The core container has no Docker socket or GPU passthrough, and doesn't
include all upstream harness dependencies. For the code and agent adapters,
prepare a dedicated host worker with pinned interpreters, source checkouts and
sandbox images:

| Command or flag | Does |
| --- | --- |
| `vllm-sr benchmark setup --benchmark all` | Inspects the prerequisites |
| `--install` | Installs the pinned optional environments, and fetches SciCode test data with a verified SHA256 |
| `--build-sandbox` | Builds the offline code-grading image; its receipt records the image and base-image digests and the pinned dependencies |

Neither setup operation calls a model.

- **Cache.** The default cache is `~/.cache/vllm-sr/sr-bench-1.0`, overridable
  with `SR_BENCH_HOME`. SciCode data defaults to
  `assets/scicode/test_data.h5` inside it; `SR_BENCH_SCICODE_TEST_DATA` can
  select another prepared file.
- **Still needed:** the declared preparation for terminal task images, source
  access, judges and simulators.
- **Use it.** Configure `SR_BENCH_URL` to use that worker instead of creating
  the local container. The URL must be reachable from each client; a
  host-local address and a Dashboard-container address can differ while
  referring to the same service.

## A standalone service for development

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

- **Binding beyond loopback** requires `SR_BENCH_TOKEN`; `SR_BENCH_TOKEN_ENV`
  can name a custom service credential. Use authenticated, private deployment
  wiring: browser users reach the Dashboard proxy, not the worker directly.
- **Model credentials** use separate `api_key_env` references. Keep their
  values out of manifests, command arguments and public artifacts.

## Next

- [Prepare reusable tasks and targets](./tasks-and-targets.md)
