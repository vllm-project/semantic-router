# OpenShift demo helpers

These scripts send sample requests to an existing OpenShift deployment and
make Router logs easier to follow. They are useful for an interactive demo or
diagnosis; they do not measure classification accuracy, latency, cache benefit,
or production readiness.

## Requirements

- `oc` logged in to the target cluster;
- a completed deployment in `vllm-semantic-router-system`;
- Python 3 for the interactive client;
- permission to read Routes and Router logs.

The scripts discover current Route hosts. Do not copy hosts from saved output.

## Run a request walkthrough

In one terminal, follow selected Router events:

```bash
deploy/openshift/demo/live-semantic-router-logs.sh
```

In a second terminal, use either the interactive client or a small
classification set:

```bash
python3 deploy/openshift/demo/demo-semantic-router.py
```

```bash
deploy/openshift/demo/curl-examples.sh all
```

The log highlighter depends on current log message shapes. When it shows no
event, inspect the raw log before assuming the Router did not run:

```bash
oc logs deployment/semantic-router \
  --namespace vllm-semantic-router-system \
  --tail=100
```

Request bodies, classifications, and security decisions can contain sensitive
data. Do not use the live-log scripts in a tenant environment without an
approved logging and redaction policy.

## Cache helper

```bash
deploy/openshift/demo/cache-management.sh status
```

The helper's `clear` action restarts the Router to clear process-local state.
That interrupts traffic and does not clear an external cache backend. Use it
only in a disposable demo deployment:

```bash
deploy/openshift/demo/cache-management.sh clear
```

Do not derive a cache speedup claim from two interactive requests. Use the
benchmark tools under [`bench/`](../../../bench/) with controlled inputs and a
recorded backend when performance evidence is required.

## Files

| File | Purpose |
| --- | --- |
| `demo-semantic-router.py` | Menu-driven routing and policy requests. |
| `curl-examples.sh` | Direct classification examples. |
| `live-semantic-router-logs.sh` | Highlights Router pipeline events. |
| `live-classifier-logs.sh` | Focuses on classifier API logs. |
| `cache-management.sh` | Shows or clears demo cache state. |
| [`CATEGORY-MODEL-MAPPING.md`](CATEGORY-MODEL-MAPPING.md) | Current demo decision-to-model mapping. |

The JSON results file is captured output, not a maintained acceptance result.
