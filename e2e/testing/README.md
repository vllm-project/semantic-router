# Specialized and legacy test utilities

This directory contains manual Python integration scripts and supporting test
servers that predate or sit outside the Go E2E profile runner.

For maintained Kubernetes profile coverage, start with
[`e2e/README.md`](../README.md) and `make e2e-test`. Do not infer current CI
coverage from a numbered script or a saved check mark in this directory.

## Contents

- [`llm-katan/`](llm-katan/) provides a lightweight OpenAI-compatible test
  server and echo backend.
- [`anthropic-shim/`](anthropic-shim/) translates a llama.cpp-style backend to
  the Anthropic Messages shape for a manual profile.
- [`hallucination-demo/`](hallucination-demo/) runs a focused mock-tool demo.
- [`vllm-sr-cli/`](vllm-sr-cli/) documents CLI unit and container integration
  tests.
- `run_response_api_suite.sh` drives the manual Responses API storage matrix.
- numbered Python scripts probe individual API or routing paths and may require
  a separately started Router, Envoy, or provider account.

## Local manual stack

Use this only for a script that explicitly requires it:

```bash
# Terminal 1: lightweight model backends
e2e/testing/start-llm-katan.sh

# Terminal 2: Envoy
make run-envoy

# Terminal 3: Router with the E2E config
make run-router-e2e
```

Then run the named script, for example:

```bash
python3 e2e/testing/00-client-request-test.py
```

Read the script before running it. Ports, credentials, dependencies, and
assertion strength vary; some utilities are diagnostic rather than blocking.

`09-openai-api-validation-test.py` sends requests to an external provider and
requires `OPENAI_API_KEY`. It may incur cost and transfer test content outside
the local environment.

## Adding coverage

Prefer a registered Go test case and profile when the behavior is a supported
Kubernetes contract. Keep a standalone script only for a distinct runtime,
provider, or manual diagnostic, and state its prerequisites and pass/fail
condition beside it.
