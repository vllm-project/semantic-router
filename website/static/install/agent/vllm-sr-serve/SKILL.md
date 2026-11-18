---
name: vllm-sr-serve
description: Start and verify a vLLM Semantic Router v0.5 local stack with vllm-sr serve, or serve a decision model in Engine mode. Use when someone asks how to run vLLM-SR, chooses a CPU or GPU platform, has a model endpoint or artifact, or needs a safe first-start and readiness check.
---

# Serve vLLM Semantic Router

Use the installed CLI and the maintained
[deployment procedure](https://vllm-sr.ai/install/agent/vllm-sr/references/deployment-loop.md)
for the exact platform, configuration, and recovery steps. This skill covers
the local-image `vllm-sr serve` path. For Kubernetes, upgrades, rollback,
and removal, use the [operations skill](https://vllm-sr.ai/install/agent/vllm-sr/SKILL.md).

The **decision model** answers Router signal questions. The **backend model**
answers the user's request. Router mode needs a reachable backend and a
canonical configuration; Engine mode serves a decision-model artifact through
System One without recipe routing. Do not present an Engine response as a
routed answer.

## 1. Inspect before starting

Run the [read-only preflight](https://vllm-sr.ai/install/agent/vllm-sr/SKILL.md#preflight).
Record the CLI version, `vllm-sr config schema` version, Docker or Podman
access, CPU/GPU devices, free memory and disk, occupied ports, an existing
config, and `vllm-sr status`. Select only a supported platform from the
[deployment matrix](https://vllm-sr.ai/docs/installation/support-matrix).

```bash
vllm-sr --version
vllm-sr config schema
vllm-sr status
```

The skill follows v0.5 config. If the installed schema is older, use the
[CLI installation step](https://vllm-sr.ai/install/agent/vllm-sr/references/deployment-loop.md#2-install-the-cli)
and report the chosen release channel. If a stack is already running, verify
it in step 4; do not call `serve` again, because that restarts its containers.
Do not stop an unrelated stack or publish a listener beyond loopback without
the user's authorization. Keep credentials in environment variables.

## 2. Choose one mode

| Goal | Required input | First maintained path |
| --- | --- | --- |
| Route user requests | A complete canonical config and an answer-model endpoint reachable from the Router container | [Write and validate the config](https://vllm-sr.ai/install/agent/vllm-sr/references/deployment-loop.md#3-write-the-configuration), then Router mode below |
| Ask a decision model directly | A supported decision-model artifact and enough CPU RAM or an available GPU | [Engine mode](https://vllm-sr.ai/docs/model-runtime/get-started/quickstart), then the System One check below |

For Router mode, inspect an existing `config.yaml` rather than replacing it.
The [deployment reference](https://vllm-sr.ai/install/agent/vllm-sr/references/deployment-loop.md#3-write-the-configuration)
has the maintained first-stack example and the container-side endpoint check.
It binds the inference listener to loopback and includes a model-backed signal
so readiness can prove the decision model loaded. Validate the exact file:

```bash
vllm-sr config validate --config config.yaml
```

Pass: exit 0 with no validation errors. An incomplete config or unreachable
backend is a stop condition; report the failing field or endpoint. Do not
start bare `vllm-sr serve` in Router mode: without a complete config it can
enter interactive Dashboard setup and wait indefinitely.

## 3. Start the chosen mode

Run from the chosen config directory so stack state stays beside it. For a
CPU Router stack:

```bash
vllm-sr serve --config config.yaml --platform cpu
```

Pass: exit 0 and `vLLM Semantic Router is running` with the gateway and
endpoint summary. The first start may download images and the decision model;
allow the documented startup budget and preserve its output. On a validated
AMD or NVIDIA host, use `--platform rocm` or `--platform cuda` after checking
the [GPU requirements](https://vllm-sr.ai/install/agent/vllm-sr/references/deployment-loop.md#gpu-platforms).
Use `--gateway extproc` only when the deployment requires Envoy. For an
existing running stack, stay in verification; a config change follows the
[configuration workflow](https://vllm-sr.ai/install/agent/vllm-sr/references/configuration-loop.md#change-a-running-stack).

For a standalone decision-model trial on the CPU, use Engine mode:

```bash
vllm-sr serve vllm-sr/vela3-fast --engine --platform cpu
```

Pass: exit 0 and a `Ready:` line naming the System One URL. A GPU path may
use `--platform rocm` or `--platform cuda` and explicit `--device-ids` as
shown in the [Engine quickstart](https://vllm-sr.ai/docs/model-runtime/get-started/quickstart).
Do not switch modes on a running stack without reviewing its current config
and stopping point. A failed download, startup timeout, or occupied port needs
the [matching troubleshooting check](https://vllm-sr.ai/install/agent/vllm-sr/references/troubleshooting.md),
then a retry of the same validated command; do not silently change the image,
model, or configuration.

## 4. Prove the result

For Router mode, follow the
[routing verification procedure](https://vllm-sr.ai/install/agent/vllm-sr/references/route-verification.md).
Check `vllm-sr status`, the model-runtime inventory, a traced preview, and a
bounded end-to-end probe using the actual entrypoint, decision, and backend
model from the config. Preview proves selection without generation; probe
proves that the selected backend delivered an answer.

```bash
vllm-sr status
vllm-sr route preview --model vllm-sr/auto --prompt 'Define a readiness probe in one sentence.' --trace --json
vllm-sr route probe --config config.yaml --model vllm-sr/auto --prompt 'Define a readiness probe in one sentence.' --max-completion-tokens 256
```

Pass only when status says Router running, required deployments are `ready`
in `GET /api/v1/inventory/model-runtime`, preview selects the expected
decision/model with no signal error, and probe exits 0 with a complete answer
and matching `x-vsr-selected-decision`, `x-vsr-selected-model`, and
`x-vsr-response-path` headers. A probe that reaches its token cap or selects
fallback is not proof of the intended route. Record the config identity and
any difference between preview and probe.

For Engine mode, `GET /v1/systemone/models` must list the selected artifact.
Ask it one bounded question through `POST /v1/systemone`:

```bash
vllm-sr status
curl -fsS http://127.0.0.1:8899/v1/systemone/models
curl -fsS http://127.0.0.1:8899/v1/systemone \
  -H 'content-type: application/json' \
  -d '{"model":"vllm-sr/vela3-fast","state":"Our export fails with a 500 and blocks work.","questions":{"urgent":{"type":"noul","instructions":"Does the user report a problem that blocks their work?"}}}'
```

Pass: the model list contains `vllm-sr/vela3-fast`, and the response has an
`answers.urgent.noul` number. An HTTP 200 with no answer is not success.
If the listener has API keys, use its documented credential environment
variable without placing a key in the request text or report. Other models
or ports require replacing the example values from the actual listener and
[System One API](https://vllm-sr.ai/docs/system-one/use/api).

## 5. Hand off

Report mode, CLI version/channel, platform, config or artifact identity,
stack directory, published endpoints, observed readiness, preview/probe or
System One evidence, and any missing check. Name the exact stopped step and
retry path when evidence is incomplete. Link the
[maintenance procedure](https://vllm-sr.ai/install/agent/vllm-sr/references/maintenance.md)
for an authorized stop, rollback, or teardown.
