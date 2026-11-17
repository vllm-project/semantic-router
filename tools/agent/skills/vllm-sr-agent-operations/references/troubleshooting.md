# Troubleshooting

Find the symptom, run its check, apply its fix, and rerun the step that failed
([main skill](https://vllm-sr.ai/install/agent/vllm-sr/SKILL.md)). Change one thing at a time. `vllm-sr logs router` and
`vllm-sr logs dashboard` print the last 200 lines (`-f` follows);
`vllm-sr status` reports what runs and whether setup or a restart is pending.
`ROUTER_ORIGIN` is the stack's management origin.

## Install

| Symptom | Check | Fix |
| --- | --- | --- |
| `The virtual environment was not created successfully because ensurepip is not available` | `python3 -m ensurepip --version` fails | Ubuntu or Debian without `python3-venv`: with the user's OK, `sudo apt-get install -y python3-venv`, then rerun the installer. |
| `vllm-sr: command not found` after the install | `ls ~/.local/bin/vllm-sr` | `export PATH="$HOME/.local/bin:$PATH"`. |
| `vllm-sr config schema` reports `v0.3`, or fails | `vllm-sr --version` | The release predates the v0.5 configuration. Install `--channel dev` ([Deployment](https://vllm-sr.ai/install/agent/vllm-sr/references/deployment-loop.md#2-install-the-cli)). |
| `pip install --pre vllm-sr` installs a release, not a dev build | `vllm-sr --version` | Dev versions sort below the release they follow ([#4695](https://github.com/vllm-project/semantic-router/issues/4695)). Use the installer's `--channel dev`. |
| `permission denied while trying to connect to the Docker daemon socket` | `id -nG` lacks `docker` | Ask the user to add the account (`sudo usermod -aG docker "$USER"`, then a new login). Don't run `vllm-sr` with `sudo`: its state would belong to root. |

## Start

| Symptom | Check | Fix |
| --- | --- | --- |
| `serve` prints `Waiting for setup` and doesn't return | `vllm-sr status` says `Setup mode` | No complete `config.yaml`. Stop waiting (Ctrl-C), run `vllm-sr stop`, write the configuration (setup mode writes its own `config.yaml` there; replace it), and serve with `--config config.yaml`. |
| `… port N is already in use` | `docker ps`; `ss -ltnp` | Ask the user before stopping anything, or start this stack with a `VLLM_SR_PORT_OFFSET` ([stack identity](https://vllm-sr.ai/install/agent/vllm-sr/references/deployment-loop.md#stack-identity)). |
| `Error: Invalid runtime config: the Router in IMAGE refuses it: …` | IMAGE is the Router `serve` was about to start; the running stack is untouched | Fix the field the message names, check it with `vllm-sr config validate --image IMAGE`, and serve again. IMAGE is a GPU image you didn't want: on a GPU host `auto` picks the GPU platform, so pass `--platform cpu`. The message ends with `` `latest` is a moving tag: … ``: the local copy is older than the CLI; run the rebuild it names, or name the image you mean with `--image` ([Run a specific image](https://vllm-sr.ai/install/agent/vllm-sr/references/deployment-loop.md#run-a-specific-image)). |
| `Failed to pull image: IMAGE` | `docker pull IMAGE` | Fix registry access. An image that only exists on this host needs `--image-pull-policy ifnotpresent`. A failed pull leaves a running stack untouched. |
| `serve` times out | `vllm-sr logs router`; the last `Downloading model files` or `Waiting for Router-managed model deployments` line | Image pulls, the model download or GPU compilation took longer. Rerun with `--startup-timeout 3600`; the containers stay up after a timeout. |
| `serve` exits 1 with `Instance controller failed to start: …; its log is PATH` | The message's own reason; `tail PATH` | `Instance socket address … is longer than the 103 bytes a unix socket allows`: export `XDG_RUNTIME_DIR` as a shorter directory you own with mode 700, and serve again. `… must be owned by the current user: /tmp/vllm-sr-UID`: another account created that directory; ask the user to remove it, or export `XDG_RUNTIME_DIR` as above. |
| `Platform 'amd' selected but missing AMD GPU devices` | `ls -l /dev/kfd /dev/dri` | The ROCm driver isn't loaded on the host, or the devices aren't visible. |
| NVIDIA: `could not select device driver "" with capabilities: [[gpu]]` | `docker info` lists no `nvidia` runtime | The NVIDIA Container Toolkit is missing; ask the user to install it. |
| `--platform rocm/cuda needs a Linux host` | macOS | Serve without `--platform`; the Docker target runs on the CPU. |

## Configuration and migration

| Symptom | Check | Fix |
| --- | --- | --- |
| A one-line error with `removed_signal_type`, `removed_field`, `removed_algorithm_type`, `removed_plugin_type`, `removed_section` or `invalid_value`, ending in ``run `vllm-sr config migrate` (…)`` | The line names the path, the form and its replacement | Migrate ([Maintenance](https://vllm-sr.ai/install/agent/vllm-sr/references/maintenance.md#migrate-a-configuration-to-v05)), review the printed changes, validate. |
| `version: invalid_value: config schema v0.3 is removed` | The file says `version: v0.3` | A v0.4-era configuration: migrate it. |
| `Error: Output file already exists: config.migrated.yaml` | `ls config.migrated.yaml` | Rerun with `--force`; the same input gives the same output. |
| `No Router could migrate the configuration: …` | The Router migrate asks isn't on the host: `VLLM_SR_ROUTER_BINARY`, else the image `VLLM_SR_ROUTER_IMAGE` or `VLLM_SR_IMAGE` names, else this CLI's own release image | Run the command the message names (`docker pull IMAGE`, or `make -C REPO vllm-sr-build` from a source checkout), or export `VLLM_SR_IMAGE` as an image the host has; then migrate again. Migrate never pulls, never asks a running stack and has no `--image` option. |
| Migrate or validate refuses a form this CLI writes, and the message ends with `` `latest` is a moving tag: this local copy was built on DATE and may be older than this CLI; … `` | `docker image ls 'ghcr.io/vllm-project/semantic-router/*'` | A development CLI runs the moving `latest` images, and the host's copy predates it. Run the pull or rebuild the message names, or name the image you mean (`VLLM_SR_IMAGE`; validate also takes `--image`), then rerun. |
| `it has no mechanical rewrite: …` | The path and form before it: the `image_gen` plugin, or an audio or image-anchor `embeddings` rule | Edit a copy as the message says, then migrate again. |
| A route, retrieval or rule of the old configuration is gone after migrate | Migrate's `Changes to review` and `Warning:` lines (keep its output) | Migrate drops what v0.5 can't execute (`kb` rules, a text-and-image route over two models, an enabled `mcp` RAG source) and says how to restore each; see [Maintenance](https://vllm-sr.ai/install/agent/vllm-sr/references/maintenance.md#what-migrate-does-with-v04-forms). Restore what the user needs in the migrated file, then validate again. |
| `Only the CLI's structural checks ran: …` | No Router image on the host, or `--offline` | Run the command the line names, pass `--image IMAGE` for one the host has, or `--endpoint "$ROUTER_ORIGIN"` to ask the running Router. |
| `This stack's running Router is a different image: IMAGE.` after `Checked by …` | Which image checked the file | Information, not an error: validate checks with the image `serve` would run (`--image`, `VLLM_SR_ROUTER_IMAGE`, `VLLM_SR_IMAGE`, else this CLI's own), never the running stack's. |
| `….compress.signals: the routing signals read the request before a decision is chosen; …` | A decision's own `context` plugin sets `compress.signals` | Move `signals` to the `context` plugin of `global.plugins` or of the recipe ([Configuration](https://vllm-sr.ai/install/agent/vllm-sr/references/configuration-loop.md#selectors-executors-and-plugins)). |
| `span_required` | A `pii`, `hallucination`, `span` or masking rule resolves to a deployment without spans (Decision 1.0 to 3.0) | Bind those signals to a span-capable deployment: `global.model_catalog.bindings.pii: primary`. |
| `question_type_unsupported: resolves to deployment "D", which answers … questions and cannot answer TYPE questions; …` | What D's model answers ([Configuration](https://vllm-sr.ai/install/agent/vllm-sr/references/configuration-loop.md#signals-are-questions-to-the-decision-model)): vela3-fast, the default, answers no `score`; Decision 1.0 to 3.0 answer no `span` | Bind the rule or its signal type where the message says to a deployment that answers TYPE, such as a vela3-lite or Vela 2.0 deployment for `score`, or ask a type D answers. A judge, selector or answer check says instead `deployment "D" serves a model that answers … questions, so it does not answer a TYPE question`. |
| `span_head_unsupported: resolves to deployment "D", whose model has span heads … and no broad head; …` | The rule's `head` | Only Vela 2.0 0.8B, 4B and 9B have the `broad` head. Drop `head`, or bind the rule to such a deployment. |
| `input_modality_unsupported` | A rule reads `input.image`, `video` or `audio` on a model that doesn't | Bind the rule to a model that reads the medium (the d3 family for images), or drop the input. |
| `uncalibrated_preset` (a warning) | The rule has no `predicate` and its model no calibrated threshold | Expected for remote and d3 models, which use 0.5. Write a `predicate` from your own measurements. |
| `match_empty` | A `match` block has neither `recall` nor `judge` | Add a stage. |
| `renamed_signal_type`, `renamed_algorithm_type` (warnings) | `decision` used as a signal or algorithm type | It loads as `systemone`; migrate rewrites it. |
| `… needs an identity source in standalone mode; …`, from migrate (with ``Hint: Re-run with `--gateway extproc` …``), validate or serve | `debug: authenticated`, authz signals or providers, per-user keys or user rate limits read the caller's identity headers | A configuration of an earlier release ran behind Envoy: migrate, validate and serve it with `--gateway extproc`. Behind the user's own authenticating proxy, `listeners[].identity.trust_headers: true` serves it standalone; ask first, since it trusts what clients send if they can bypass the proxy. On a loopback stack, `debug: any` needs no identity. |
| `output_capability_missing`, `output_not_executable` | The decision's `output` against its models' `capabilities` and `modalities` | Declare the capability on the model; image edits, audio and video can't execute yet. |
| `….auth.credential: delegated requires global.services.authz.delegated_credential.enabled: true` | The delegated-credential contract is off, its default | Enable it only behind an authenticator that writes the header ([Configuration](https://vllm-sr.ai/install/agent/vllm-sr/references/configuration-loop.md#upstream-credentials)). |
| `….auth.credential: delegated cannot combine with a static key (…)` | The upstream also names a key | A delegated upstream takes only the user's credential: remove the key, or remove `credential`. |
| `….auth.credential: delegated, but …; a Router-own call never carries a user's credential, give it a static key` | The reason names who calls it: only Router-own calls, vector store ingestion, memory storage, `router_replay.shadow` copies, or the Router's checks of a `model_runtime` endpoint | Give that upstream a static key. |
| `… response_cache scope "…" shares answers across users, and … is delegated; …` | The plugin's `scope` | Use `scope: user`, the default. |
| `removed_field: trusted_gateway is removed` or `gateway_credential is removed` | A configuration that uses the trusted gateway credential of `main` before v0.5 | Write `auth: {credential: delegated}` and `global.services.authz.delegated_credential` by hand; migrate has no rewrite for them. |

## Decision model and embeddings

| Symptom | Check | Fix |
| --- | --- | --- |
| A request that a decision-model rule should match takes a lower decision | `vllm-sr route preview --model vllm-sr/auto --prompt PROMPT --trace --json` shows `signal_errors`, such as `"domain:computer_science": "systemone_failed"` | The deployment didn't answer. Check it below; the rule resolves through `on_unknown` until it does. |
| The local decision model isn't ready | `curl -s "$ROUTER_ORIGIN/api/v1/inventory/model-runtime"`: `"ready": false` with `state` `loading` or `warming` | It is still downloading or warming. Wait; give `config apply` `--timeout 900`. Check free disk and memory. |
| `"state": "incompatible"` with `the model runtime serves contract X; this router was built for 4.0.0` | An attached `model_runtime` deployment with an `endpoint` | The Router and its runtime must come from one release; upgrade the runtime. |
| `"state": "failed"`, or `restarts` keeps growing | `vllm-sr logs router`; `docker stats` | Out of memory or a missing device. Give the stack memory, move the model to a GPU, or choose a smaller one ([Optimization](https://vllm-sr.ai/install/agent/vllm-sr/references/optimization.md#choose-the-decision-model)). |
| `{"deployments": []}` | The configuration has no decision-model rule | Expected: nothing asks the model, so none starts. |
| A remote decision model never answers | `vllm-sr logs router` shows `systemone_deployment_failed` with the reason | Fix the deployment's `endpoint`, `model` or `auth.api_key_env`, or raise its `timeout_ms`. |
| `embedding provider was not prepared`, or `EMBEDDING_UNAVAILABLE` | The inventory's `vela-embedding`, or the remote `openai_embeddings` deployment | The recall deployment isn't ready. Wait for it, fix the remote one, or recall with `recall: {method: bm25}`. |
| The load refuses a recall deployment | Its `capabilities.dimensions` and `modalities` | A recall deployment must embed text, in the dimension the store's `embedding` declares. |

## Requests and headers

| Symptom | Check | Fix |
| --- | --- | --- |
| Status `502`, `503` or `504`, or the request hangs | The backend from a container: `docker run --rm --add-host=host.docker.internal:host-gateway curlimages/curl:8.12.1 -s -m 5 URL` | A server on the host must listen beyond 127.0.0.1 and be addressed as `host.docker.internal:PORT`. `vllm-sr logs router` shows the upstream error. |
| `route probe`: `Read timed out. (read timeout=120.0)` | The backend is still generating | Pass `--max-completion-tokens 256` with a prompt that asks for a short answer, and `--timeout` for slow hosts. |
| `route probe` exits 2 with `Completion was truncated at the token or context limit` | `finish_reason: length` in the receipt | Raise `--max-completion-tokens` or ask for a shorter answer. |
| `x-vsr-selected-model` names an unexpected model, or no `x-vsr-*` headers | The request's `model` | A concrete model name bypasses routing. Use `vllm-sr/auto` or an entrypoint from `GET /v1/models`, sent to the Router's listener. |
| No `x-vsr-trace` header | The request sent `x-vsr-debug: true`; `global.router.headers.debug` | It is `none` by default. Set `debug: any` on a loopback or test stack, or `authenticated` with `debug_principals` behind a gateway. |
| `x-vsr-matched-*`, `x-vsr-cache-hit` or `x-selected-model` is missing | v0.5 removed them | Read the trace's `signals`, `x-vsr-response-path` (`cache`) and `x-vsr-route-model`. |
| A policy that reads the user never matches in standalone mode | The listener's `identity.trust_headers` | Client identity headers are dropped unless the listener trusts them; trust them only behind an authenticating proxy. |
| `401` with `delegated_credential_missing`: `The selected upstream requires a delegated credential, which the request does not carry.` | Whether the request came through the authenticator, and the header `global.services.authz.delegated_credential.header` names (default `x-vsr-delegated-credential`) | The authenticator must write that header and the user's id on every request; client copies are stripped. Nothing was sent upstream. |
| `400` with `"code":"image_route_required"` | `POST /v1/images/generations` | No decision answers with an image; add one with `output: {kind: image, action: generate}` ([Configuration](https://vllm-sr.ai/install/agent/vllm-sr/references/configuration-loop.md#output-contracts-and-images)). |
| `unsupported_capability` from Chat Completions or Messages | The matched decision's output is an image | Use the Responses API or the Images API for image decisions. |
| The first token is slow only on guarded decisions | The decision's `response_guard` has a `block` rule, an `annotate` rule with `details`, or `on_unknown: block`; `llm_response_guard_buffered_total` on the metrics port counts that decision | Expected: the Router holds such a decision's streamed answer until the guard has read all of it, then sends it, so the first token arrives after the whole answer. A guard that only records (`none`, or `annotate` without `details`) keeps live streaming ([Optimization](https://vllm-sr.ai/install/agent/vllm-sr/references/optimization.md#latency)). |
| A streamed answer of a guarded decision is HTTP 200 with only a stream error (a Chat `error` chunk, a Responses `response.failed`, a Messages `error` event) | `llm_response_guard_buffered_total{decision_name="D"}` by `outcome`; with trace access, the outcome in the `response_guard` member of `x-vsr-trace` | `blocked`: the guard refused it, as it would refuse an unstreamed answer with 403. `over_limit`: the held answer passed 64 MiB and fails closed. `incomplete`: the model's stream ended early. Behind Envoy, Envoy's buffer and ExtProc message limits also fail closed, with 500. |
| Guarded decisions hit the response cache less often | The decision's `response_guard` | Expected: the cache stores only answers the guard released; blocked, warnings-only, over-limit and incomplete answers are never stored, and a change to the guard starts its cache fresh. |
| A semantic cache hit on a guarded decision serves an answer the guard would have flagged for this request | The guard's conditions on the request (a `keyword` or `fact_check` condition, for example) | A semantic hit serves the answer released for a similar request as is: the guard doesn't run on hits. Use `mode: exact` for that decision's cache. After changing the threshold of a signal the guard reads, set a new `revision.policy_revision` in its `response_cache` configuration rather than flushing the store ([Optimization](https://vllm-sr.ai/install/agent/vllm-sr/references/optimization.md#caches-and-judge-caps)). |
| `vllm-sr logs envoy`: `This stack runs in standalone mode, with no Envoy container` | `serve` printed `standalone gateway` | Expected. Envoy runs only with `--gateway extproc`. |

## Changes to a running stack

| Symptom | Check | Fix |
| --- | --- | --- |
| Editing `config.yaml` changes nothing | The stack serves `.vllm-sr/runtime-config.yaml` | Apply it: `vllm-sr config apply --config config.yaml` ([Configuration](https://vllm-sr.ai/install/agent/vllm-sr/references/configuration-loop.md#change-a-running-stack)). |
| `config apply`: `Restart required: run vllm-sr serve to apply.` | The change touches a listener (with `--gateway extproc`, also the provider topology) | It is saved; `vllm-sr serve --config config.yaml` applies it. |
| `config apply`: `Router management API request timed out after 120s` | `vllm-sr config versions` | The change can still activate, for example while a model loads. Check before retrying; pass `--timeout 900` next time. |
| `serve` warns `Preserving Dashboard or package changes in runtime config` | The active document differs from `config.yaml` | Dashboard or API edits are active. Keep them, or ask and serve with `--replace-active-config`. |
| `status`: `Restart required` | A saved change waits | `vllm-sr serve --config config.yaml`. |

## sr-bench

| Symptom | Check | Fix |
| --- | --- | --- |
| A MoM target fails with `Frozen runtime identity verification failed` | `global.router.headers.debug` | The run's receipts ride in `x-vsr-trace`. Allow the worker's requests (`debug: any` on a loopback stack), then plan again. |
| `Frozen output or call limit exceeded`; the call shows `Output token cap exceeded` | `vllm-sr benchmark show RUN_ID --call-id CALL_ID` | The answer was longer than `limits.max_output_tokens`. Raise it and `sampling.max_tokens` in a new plan. |
| A target is unreachable from the worker | The worker runs in the stack's sr-bench container | Use `http://host.docker.internal:PORT/v1` for a server on the host and `http://vllm-sr-router-container:8899/v1` for the stack's own listener (`NAME-vllm-sr-router-container` for a named stack). |
| MoM calls fail with 412 | The target's `config_hash` | It must be the active hash: `curl -s "$ROUTER_ORIGIN/api/v1/config/hash"` → `active_runtime_hash`. Register it and plan again. |

## Kubernetes

| Symptom | Check | Fix |
| --- | --- | --- |
| `Helm chart directory not found` | `--chart-dir`, or `./deploy/helm/semantic-router` | A dev CLI looks for the published chart or a local one; pull the chart there or pass `--chart-dir`. |
| `'helm' is required for Kubernetes deployment` | `helm version` | Install `helm` 3 and `kubectl`, with the user's OK. |
| `Kubernetes deployment does not support Dashboard setup-mode configs` | The config is incomplete | Write a complete configuration first. |
| Pods run, requests fail upstream | The backend address | It must resolve inside the cluster; `host.docker.internal` doesn't. |
| The operator refuses a `SemanticRouter` with `spec.config.routing` | The resource | Rewrite it as `spec.config.recipes` (and `entrypoints`, `plugins`, `models`). |
