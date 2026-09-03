# Hallucination detection demo fixtures

This directory contains deterministic mock services and clients for exploring
the Router's hallucination-detection flow with tool context. It is useful for
local UI and client development; the maintained pass/fail contract is the
Kubernetes `hallucination` E2E profile.

## Maintained E2E check

From the repository root:

```bash
make e2e-test E2E_PROFILE=hallucination
```

That profile deploys an OpenAI-compatible mock LLM and endpoint-backed mock
detector, sends a fact-checkable request with tool context, and verifies the
`x-vsr-response-warnings: hallucination` response header. Its configuration is
in [`../../profiles/hallucination/values.yaml`](../../profiles/hallucination/values.yaml).

## Local fixture components

| File | Role |
| --- | --- |
| `mock_vllm_toolcall.py` | Returns a tool call, then a deliberately inconsistent final answer. |
| `mock_web_search.py` | Supplies deterministic reference facts to the client. |
| `chat_client.py` | Runs the CLI tool-call loop and prints Router warning headers. |
| `web_client.py` | Provides a browser client for the same local services. |
| `run_demo.sh` | Starts the local mocks, Router binary, Envoy, and a selected client. |

The mocks cover three fixed topics: the Eiffel Tower, Tokyo population, and
Apple's founders. They test plumbing and warning propagation, not detector
quality on an open-ended dataset.

## Local prerequisites

The launcher expects:

- a Router binary and downloaded Candle models;
- Python 3 with the client dependencies used by the scripts;
- `curl`, `lsof`, and `func-e` on `PATH`;
- ports 50051, 8002, 8003, 8080, 8801, and optionally 8888 to be free.

It terminates processes using those local ports during setup and cleanup.
Use the Kubernetes E2E profile for CI or when you do not want that local
process management behavior.
