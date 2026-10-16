---
name: vllm-sr
description: Install, configure, and verify vLLM Semantic Router on a Docker host (CPU, AMD, or NVIDIA GPU) or a Kubernetes cluster, then operate it through its CLI and Router API. Use for a first install, an upgrade, routing verification, configuration changes, recipe tuning, and single-model/MoM evaluation.
---

# vLLM Semantic Router: install, configure, verify

Run the steps in order. Each one says what to run, what success looks like and
what to do otherwise, and each is safe to run again.

The Router serves an OpenAI-compatible API and picks a model for each request.
It does not run the models that answer users; those are the user's endpoints,
such as vLLM, Ollama or a hosted API. `vllm-sr serve` runs a stack in Docker:
the Router (inference on port 8899, management API on 8080), the Dashboard
(8700), the model runtime for the Router's own classifiers, and supporting
services.

## Ground rules

- Keep secrets in environment variables, never in YAML, command arguments,
  logs or your report.
- Don't stop, restart or replace what you didn't start (containers, a running
  `vllm-sr` stack, services on its ports) without asking.
- Ask before system-wide changes: installing packages, adding the user to a
  group, publishing a port beyond loopback.
- Never run `vllm-sr serve` without a complete `--config`. Without one it
  starts the Dashboard's setup mode and waits for a person indefinitely.
- An HTTP 200 alone is not success. Each check below names what the output
  must contain.

## 1. Preflight

These commands change nothing. Run them and keep the results.

```bash
uname -sm
docker info --format '{{.ServerVersion}}'
python3 -c 'import sys; print(sys.version.split()[0])'
python3 -m ensurepip --version
df -h "$HOME" | tail -1
ls /dev/kfd /dev/dri/render* 2>/dev/null
nvidia-smi -L 2>/dev/null
vllm-sr --version 2>/dev/null && vllm-sr status
```

| Check | Pass | Otherwise |
| --- | --- | --- |
| Docker | prints a version | "permission denied": ask the user to add their account to the `docker` group. Not installed: ask before installing it. Podman works too: add `--container-runtime podman` to `vllm-sr` commands. |
| Python | 3.10 or newer, and `ensurepip` prints `pip …` | Ubuntu and Debian: with the user's OK, `sudo apt-get install -y python3-venv`. Without it the installer stops at "ensurepip is not available". |
| Disk | 10 GB free; 30 GB for a GPU platform | Ask the user. |
| `vllm-sr` | not installed, or `status` says `Not running` | A stack is already running. It is the user's: find its directory (below), verify it with step 6 using its own decision and model names, and ask before changing it. |

Then check the host ports the stack publishes:

```bash
bash -c 'for p in 8899 8080 8700 8090 9190 50051 6379 3000 9090 16686; do
  (exec 3<>/dev/tcp/127.0.0.1/$p) 2>/dev/null && echo "port $p is in use"; done'
```

Nothing printed: pass. Otherwise ask the user. A second stack can run beside
the first: see [Deployment](https://vllm-sr.ai/install/agent/vllm-sr/references/deployment-loop.md#stack-identity).

A running stack's directory is the parent of the `.vllm-sr` directory its
Router mounts:

```bash
docker inspect vllm-sr-router-container \
  --format '{{range .Mounts}}{{println .Source}}{{end}}' | grep -m1 '/\.vllm-sr$'
```

## 2. Choose the path

Decide each row and tell the user what you chose.

| Decision | Default | Otherwise |
| --- | --- | --- |
| Channel | stable, when it has this skill's commands (step 3); else dev | The user names a version or channel. |
| Platform | `auto`: inspect the execution target | `/dev/kfd` and a `/dev/dri/render*` node exist: `--platform rocm` (AMD Instinct MI300X and MI325X are validated). `nvidia-smi -L` lists a GPU and `docker info` lists an `nvidia` runtime (the NVIDIA Container Toolkit): `--platform cuda` (works, not yet validated). macOS: always `cpu`. |
| Gateway | standalone (no flag): the Router serves the API itself | `--gateway extproc` puts Envoy in front. Use it only when the user needs Envoy's rate limiting, mTLS, JWT or OIDC, advanced route matching, or an Envoy-based gateway they already run. |
| Target | Docker on this machine | The user asks for a Kubernetes cluster: write the configuration (step 4), then follow [Deployment](https://vllm-sr.ai/install/agent/vllm-sr/references/deployment-loop.md#kubernetes) instead of steps 5 and 6. |
| Model | the endpoint the user gave | None given: look for one on this host (below) and confirm it with the user. None at all: ask. For a local trial, offer [Ollama](https://vllm-sr.ai/docs/installation/ollama). |

Look for a model server on this host:

```bash
curl -s -m 3 http://127.0.0.1:11434/api/tags     # Ollama: its model names
curl -s -m 3 http://127.0.0.1:8000/v1/models     # vLLM or another OpenAI-compatible server
```

The Router runs in a container and reaches a server on this host as
`host.docker.internal:PORT`, so that server must listen beyond 127.0.0.1
(Ollama: `OLLAMA_HOST=0.0.0.0:11434`; vLLM: `--host 0.0.0.0`). Check it the way
the Router will, here for Ollama:

```bash
docker run --rm --add-host=host.docker.internal:host-gateway \
  curlimages/curl:8.12.1 -s -m 5 http://host.docker.internal:11434/api/tags
```

Pass: the same model list.

## 3. Install the CLI

```bash
curl -fsSL https://vllm-sr.ai/install.sh | \
  bash -s -- --channel stable --mode cli --runtime skip --no-launch
export PATH="$HOME/.local/bin:$PATH"
vllm-sr --version
vllm-sr serve --help | grep -q -- '--data-parallel-size' && echo "current" || echo "predates this skill"
```

`--mode cli --runtime skip --no-launch` installs only the CLI, into
`~/.local/share/vllm-sr` with the launcher `~/.local/bin/vllm-sr`. Later
commands need that `PATH`; if your shell doesn't keep it between commands,
repeat the `export`. Without those three options the installer also starts a
setup-mode stack and returns, for a user who would rather connect models in
the Dashboard; then hand over its URL instead of steps 4–6.

If it printed `predates this skill`, the stable release predates this serve contract
(including `--engine` and replica placement). Install the development channel, which
these docs follow, over it:

```bash
curl -fsSL https://vllm-sr.ai/install.sh | \
  bash -s -- --channel dev --mode cli --runtime skip --no-launch
vllm-sr --version
```

Pass: `vllm-sr version: 0.4.0.devYYYYMMDDHHMMSS` or later. Tell the user it is a
development build of `main`. If the user insists on `0.4.0`, its stack always
runs Envoy and has no `--gateway`, engine mode or `--container-runtime`;
leave those out.

## 4. Write the configuration

Pick a directory for the stack, such as `~/vllm-sr`, and run every later
`vllm-sr` command from it. The stack keeps its state in `.vllm-sr/` next to
`config.yaml`.

If `config.yaml` already exists there, it is the user's: use it, or ask.
Otherwise write it, replacing `MODEL`, `provider` and `endpoint` with the
chosen model:

```yaml
version: v0.3
listeners:
  - name: http-8899
    address: 127.0.0.1    # also the host publication: 0.0.0.0 exposes the API on every interface
    port: 8899
    timeout: 300s
providers:
  defaults:
    model: MODEL
  models:
    - name: MODEL          # the name the backend serves, such as llama3.2:3b
      backend_refs:
        - name: primary
          provider: ollama                      # or vllm
          endpoint: host.docker.internal:11434  # host:port as the Router container sees it
          protocol: http
          weight: 100
routing:
  modelCards:
    - name: MODEL
  signals:
    keywords:
      - name: code_request
        operator: OR
        keywords: ["python", "function", "bug"]
        case_sensitive: false
  decisions:
    - name: code-route
      description: Code requests. Same model; it proves that signals reach decisions.
      priority: 200
      rules:
        operator: AND
        conditions:
          - type: keyword
            name: code_request
      modelRefs:
        - model: MODEL
          use_reasoning: false
    - name: default-route
      description: Everything else.
      priority: 100
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: MODEL
          use_reasoning: false
```

```bash
vllm-sr config validate --config config.yaml
```

Pass: `✓ Configuration is valid`, with `Keywords: 1` and `Decisions: 2 total`.

- Keep `address: 127.0.0.1` unless the user wants the API on the network;
  then also set the listener's `api_keys`
  ([Gateway Modes](https://vllm-sr.ai/docs/installation/gateway-modes)).
- For a hosted API, follow
  [Configure models](https://vllm-sr.ai/docs/installation/model-configuration):
  name the key's variable in `api_key_env` and export it before `serve`.
- `vllm-sr config schema --section PATH` describes any section.

## 5. Start the stack

If `vllm-sr status` already says `Running`, don't run `serve`: it restarts
every container, and it replaces a stack started from another directory. To
apply an edited `config.yaml` to a running stack, use
[Configuration](https://vllm-sr.ai/install/agent/vllm-sr/references/configuration-loop.md).

```bash
vllm-sr serve --config config.yaml                     # cpu
vllm-sr serve --config config.yaml --platform rocm      # ROCm; CUDA uses --platform cuda
```

Add `--gateway extproc` only if you chose it, and `--minimal` to run without
the Dashboard and the observability stack. `serve` returns once the Router is
ready and prints `✓ vLLM Semantic Router is running (standalone gateway)` with
its endpoints. The first start pulls about 5 GB of images, or about 22 GB with
`--platform rocm`: minutes on a fast link, longer on a slow one. It waits up to
30 minutes (`--startup-timeout SECONDS` changes that). If your shell can't wait
that long, run it in the background and poll for the exit line:

```bash
nohup sh -c 'vllm-sr serve --config config.yaml; echo "serve exit=$?"' > serve.log 2>&1 &
grep -E 'serve exit=' serve.log    # repeat until it prints; exit=0 is success
```

## 6. Verify

Every check must pass. Header names are case-insensitive, so use `grep -i`.

1. **Status.** In `vllm-sr status`, `State`, `Router` and `Dashboard` (unless
   `--minimal`) say `Running`, and no line says `Setup mode` or
   `Restart required`.
2. **Public models.** `curl -s http://127.0.0.1:8899/v1/models` lists
   `"id":"vllm-sr/auto"`.
3. **Routing preview**, which evaluates signals and decisions without
   generating an answer:

   ```bash
   vllm-sr route preview --model vllm-sr/auto --prompt 'In one sentence, why does a Python function return None?'
   vllm-sr route preview --model vllm-sr/auto --prompt 'What is the capital of France?'
   ```

   Pass: the `Decision` line says `code-route` for the first prompt and
   `default-route` for the second, and `Selected Model` is `MODEL`. Add
   `--json` for the full result, or `--trace` to see each condition.
4. **A routed answer:**

   ```bash
   curl -s -D - -o answer.json http://127.0.0.1:8899/v1/chat/completions \
     -H 'Content-Type: application/json' -H 'x-vsr-debug: true' \
     -d '{"model": "vllm-sr/auto", "max_tokens": 64,
          "messages": [{"role": "user", "content": "In one sentence, why does a Python function return None?"}]}' \
     | grep -iE '^(HTTP|x-vsr-selected-(decision|model)|x-vsr-matched|server)'
   python3 -c 'import json; c = json.load(open("answer.json"))["choices"][0]; print(c["finish_reason"], repr(c["message"]["content"][:200]))'
   ```

   Pass: `HTTP/1.1 200 OK`, `x-vsr-selected-decision: code-route`,
   `x-vsr-selected-model: MODEL`, `x-vsr-matched-keywords: code_request`, and
   non-empty content. `server: envoy` appears only with `--gateway extproc`.
5. **Probe**, the CLI's own end-to-end assertion:

   ```bash
   vllm-sr route probe --config config.yaml --model vllm-sr/auto \
     --prompt 'In one sentence, why does a Python function return None?' \
     --max-completion-tokens 256 --expect-decision code-route --expect-selected-model MODEL
   ```

   Pass: exit code 0, `"passed": true` and `"status": 200`. The probe fails an
   answer cut off at the token cap (`finish_reason: length`), so keep a prompt
   that asks for a short answer and a cap with room to spare. Without a cap it
   waits for the whole answer, and a slow backend hits its 120 s timeout.
6. **GPU path** (`--platform rocm` or `cuda`). After checking available
   capacity, use an isolated Engine stack and a fresh config directory so the
   Router stack and its public grants are unchanged:

   ```bash
   (
     probe_dir=$(mktemp -d)
     cd "$probe_dir" || exit 1
     export VLLM_SR_STACK_NAME=vllm-sr-gpu-probe VLLM_SR_PORT_OFFSET=200
     export VLLM_SR_STATE_ROOT_DIR="$probe_dir"
     vllm-sr serve vllm-sr/Decision-2.0-Kai-0.6B -e --platform rocm \
       --device-ids 0 --minimal
     vllm-sr instance models
     curl -fsS http://127.0.0.1:9099/v1/systemone -H 'content-type: application/json' -d '{
       "model": "vllm-sr/Decision-2.0-Kai-0.6B",
       "state": "Write a Python function that merges two sorted lists.",
       "questions": {"task": {"type": "choice", "instructions": "What kind of work is this?",
         "criteria": {"code": "Writing or fixing code", "chat": "Anything else"}}},
       "options": {"return_meta": true}}'
     vllm-sr stop
   )
   ```

   Pass: the inventory reports a ready replica on the chosen GPU and the
   native request succeeds with the expected task result. Use `--platform cuda`
   on NVIDIA; choose an available host ID instead of assuming GPU 0 is free.
   The initial model download and GPU compilation can take several minutes.
   Preserve the source directory, model cache, images and volumes after stopping.

## 7. Hand off

Report to the user:

- `vllm-sr --version` and the channel, platform and gateway you chose;
- the config path, and that `vllm-sr stop` stops the stack (it keeps the
  config, `.vllm-sr/`, downloaded models, volumes and images);
- the inference API, `http://127.0.0.1:8899/v1` with model `vllm-sr/auto`;
- the Dashboard, `http://127.0.0.1:8700`; from another machine, through
  `ssh -L 8700:127.0.0.1:8700 HOST`. Its first visitor creates the
  administrator account, so the user should open it now (or set
  `DASHBOARD_ADMIN_EMAIL` and `DASHBOARD_ADMIN_PASSWORD` before `serve`);
- what each check in step 6 returned, and anything you skipped or changed.

If a step fails, find the symptom in
[Troubleshooting](https://vllm-sr.ai/install/agent/vllm-sr/references/troubleshooting.md), fix one thing, rerun that
step, and report the change. `vllm-sr logs router` shows the Router's last 200
lines.

## After the install

| Task | Reference |
| --- | --- |
| GPU details, Envoy, engine mode, Kubernetes, Dashboard access, a second stack, upgrade and removal | [Deployment](https://vllm-sr.ai/install/agent/vllm-sr/references/deployment-loop.md) |
| Symptoms and fixes | [Troubleshooting](https://vllm-sr.ai/install/agent/vllm-sr/references/troubleshooting.md) |
| Change a running configuration; versions and rollback | [Configuration](https://vllm-sr.ai/install/agent/vllm-sr/references/configuration-loop.md) |
| Deeper routing and delivery checks | [Route verification](https://vllm-sr.ai/install/agent/vllm-sr/references/route-verification.md) |
| Improve signals, decisions or model selection | [Recipe tuning](https://vllm-sr.ai/install/agent/vllm-sr/references/recipe-tuning.md) |
| Compare single models and MoM with sr-bench | [sr-bench](https://vllm-sr.ai/install/agent/vllm-sr/references/sr-bench.md) |

For the full contract, use `vllm-sr --help`, command help, `vllm-sr config
schema`, and the running Router's `GET /api/v1`. The
[Quickstart](https://vllm-sr.ai/docs/installation) and
[Gateway Modes](https://vllm-sr.ai/docs/installation/gateway-modes) describe
the same stack for people.
