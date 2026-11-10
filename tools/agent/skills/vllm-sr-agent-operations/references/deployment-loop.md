# Deploy vLLM Semantic Router

Steps 1 to 5 take a host with Docker to a checked stack; run them in order after
the [main skill](https://vllm-sr.ai/install/agent/vllm-sr/SKILL.md)'s preflight. The later sections cover GPU platforms,
engine mode, Envoy, Kubernetes, Dashboard access, a second stack, a pinned
image and new backends. Installed `vllm-sr serve --help` lists every option,
grouped by where it applies.

## 1. Choose the path

Decide each row and tell the user what you chose.

| Decision | Default | Otherwise |
| --- | --- | --- |
| Channel | stable, when its CLI reports config v0.5 (step 2); else dev | The user names a version or channel. A package the user supplies, such as a wheel, installs with the installer's `--pip-spec SPEC`. |
| Platform | `auto`, which inspects the execution target | `/dev/kfd` and a `/dev/dri/render*` node exist: `--platform rocm` (AMD Instinct MI300X and MI325X are validated). `nvidia-smi -L` lists a GPU and `docker info` lists an `nvidia` runtime (the NVIDIA Container Toolkit): `--platform cuda`. The user wants the CPU on a host with GPUs: `--platform cpu`, because `auto` picks the GPU there. macOS: always the CPU. |
| Gateway | standalone (no flag): the Router serves the API itself | `--gateway extproc` puts Envoy in front. Use it only for Envoy's rate limiting, mTLS, JWT or OIDC, advanced route matching, or an Envoy-based gateway the user runs. |
| Target | Docker on this machine | A Kubernetes cluster: write the configuration (step 3), then follow [Kubernetes with Helm](https://vllm-sr.ai/install/agent/vllm-sr/references/deployment-loop.md#kubernetes-with-helm) or [the operator](https://vllm-sr.ai/install/agent/vllm-sr/references/deployment-loop.md#kubernetes-with-the-operator) instead of steps 4 and 5. |
| Model | the endpoint the user gave | None given: look for one on this host (below) and confirm it with the user. None at all: ask. For a local trial, offer [Ollama](https://vllm-sr.ai/docs/installation/ollama). |
| Decision model | the default, `primary`: vela3-fast, on a GPU when the platform has one free, else the CPU | Memory is short (preflight): ask; see [Choose the decision model](https://vllm-sr.ai/install/agent/vllm-sr/references/optimization.md#choose-the-decision-model). |

Look for a model server on this host:

```bash
curl -s -m 3 http://127.0.0.1:11434/api/tags     # Ollama: its model names
curl -s -m 3 http://127.0.0.1:8000/v1/models     # vLLM or another OpenAI-compatible server
```

The Router runs in a container and reaches a server on this host as
`host.docker.internal:PORT`, so that server must listen beyond 127.0.0.1
(Ollama: `OLLAMA_HOST=0.0.0.0:11434`; vLLM: `vllm serve MODEL --host 0.0.0.0`).
Check it the way the Router will, here for Ollama:

```bash
docker run --rm --add-host=host.docker.internal:host-gateway \
  curlimages/curl:8.12.1 -s -m 5 http://host.docker.internal:11434/api/tags
```

Pass: the same model list.

## 2. Install the CLI

```bash
curl -fsSL https://vllm-sr.ai/install.sh | \
  bash -s -- --channel stable --mode cli --runtime skip --no-launch
export PATH="$HOME/.local/bin:$PATH"
vllm-sr --version
vllm-sr config schema | python3 -c 'import json, sys; print(json.load(sys.stdin)["config_version"])'
```

`--mode cli --runtime skip --no-launch` installs only the CLI, into
`~/.local/share/vllm-sr` with the launcher `~/.local/bin/vllm-sr`. Later
commands need that `PATH`; if your shell doesn't keep it between commands,
repeat the `export`. Without those three options the installer also starts a
setup-mode stack and returns.

Pass: the last line is `v0.5`. Anything else (`v0.3`, or an error) means the
stable release predates the configuration this skill writes. Install the
development channel, which these pages follow, over it, and run the check
again:

```bash
curl -fsSL https://vllm-sr.ai/install.sh | \
  bash -s -- --channel dev --mode cli --runtime skip --no-launch
vllm-sr config schema | python3 -c 'import json, sys; print(json.load(sys.stdin)["config_version"])'
```

Tell the user when you installed a development build of `main`.

## 3. Write the configuration

Pick a directory for the stack, such as `~/vllm-sr`, and run every later
`vllm-sr` command from it. If `config.yaml` already exists there, it is the
user's: use it, or ask. Otherwise write it, replacing `MODEL`, `provider` and
`endpoint` with the chosen model:

```yaml
version: v0.5
listeners:
  - name: http-8899
    address: 127.0.0.1 # also the host publication: 0.0.0.0 exposes the API on every interface
    port: 8899
    timeout: 300s
providers:
  defaults:
    model: MODEL
  models:
    - name: MODEL # the name the backend serves, such as llama3.2:3b
      backend_refs:
        - name: primary
          provider: ollama # or vllm
          endpoint: host.docker.internal:11434 # host:port as the Router container sees it
          protocol: http
          weight: 100
recipes:
  - name: default
    signals:
      keywords:
        - name: code_request
          operator: OR
          keywords: ["python", "function", "bug"]
          case_sensitive: false
      domains:
        - name: computer_science
          labels: ["computer science"]
    decisions:
      - name: code-route
        description: Requests that name code. A keyword rule, so no model judges it.
        priority: 200
        rules:
          operator: AND
          conditions:
            - type: keyword
              name: code_request
        modelRefs:
          - model: MODEL
            use_reasoning: false
      - name: cs-route
        description: Computer-science questions. The decision model judges the domain.
        priority: 150
        rules:
          operator: AND
          conditions:
            - type: domain
              name: computer_science
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

The three decisions use one model; they prove that a deterministic signal and
a decision-model signal both reach decisions. The `domains` rule is a question
to the decision model, which starts only because a rule asks it. Without the
memory for it, leave out the `domains` rule and `cs-route`: the stack then
loads no model of its own.

```bash
vllm-sr config validate --config config.yaml
```

Pass: exit 0 and the first line `✓ Configuration is valid`. The next line says
who checked: `Checked by the Router in IMAGE as well as the CLI.`, or, before
the first `serve` has pulled a Router image, `Only the CLI's structural checks
ran: no Router image of this CLI's version is present locally; …`. That is
enough here: `serve` pulls the image and checks the configuration with it
before it starts the stack. Validation never pulls, and it uses the image
`serve` runs by default, never a running stack's.

A development CLI runs the moving `latest` images. When the host has an older
copy, validate and `serve` refuse a configuration this CLI writes with a line
that ends in `` `latest` is a moving tag: this local copy was built on DATE
and may be older than this CLI; … ``. Run the pull or the rebuild it names, or
name the image you mean ([Run a specific image](https://vllm-sr.ai/install/agent/vllm-sr/references/deployment-loop.md#run-a-specific-image)),
then validate again.

- Keep `address: 127.0.0.1` unless the user wants the API on the network;
  then also set the listener's `api_keys`
  ([Gateway Modes](https://vllm-sr.ai/docs/installation/gateway-modes)).
- For a hosted API, follow
  [Configure models](https://vllm-sr.ai/docs/installation/model-configuration):
  name the key's variable in `auth: {api_key_env: VAR}` and export it before
  `serve`.
- `vllm-sr config schema --section PATH` describes any section.

## 4. Start the stack

If `vllm-sr status` already says `Running`, don't run `serve`: it restarts
every container, and it replaces a stack started from another directory. To
apply an edited `config.yaml` to a running stack, use
[Configuration](https://vllm-sr.ai/install/agent/vllm-sr/references/configuration-loop.md).

```bash
vllm-sr serve --config config.yaml                      # cpu, or auto
vllm-sr serve --config config.yaml --platform rocm      # ROCm; CUDA uses --platform cuda
```

Add `--platform cpu` on a GPU host the user keeps for other work,
`--gateway extproc` only if you chose it, and `--minimal` to run without the
Dashboard, sr-bench and the observability stack. `serve` returns once the
Router and its decision model are ready and prints
`✓ vLLM Semantic Router is running (standalone gateway)` with its endpoints.
The first start pulls about 4 GB of images (about 22 GB with
`--platform rocm`) and downloads the decision model (about 6 GB), printing
`Downloading model files` and `Waiting for Router-managed model deployments`
lines meanwhile. It waits up to 30 minutes (`--startup-timeout SECONDS`
changes that). If your shell can't wait that long, run it in the background
and poll for the exit line:

```bash
nohup sh -c 'vllm-sr serve --config config.yaml; echo "serve exit=$?"' > serve.log 2>&1 &
grep -E 'serve exit=' serve.log    # repeat until it prints; exit=0 is success
```

## 5. Check the stack

Set `ROUTER_ORIGIN=http://127.0.0.1:8080` (plus the port offset). Every check
must pass; [Verification](https://vllm-sr.ai/install/agent/vllm-sr/references/route-verification.md) explains each one further.

1. **Status.** `vllm-sr status` prints `State  Running`, `Router  Running`, a
   `Decision model` line and, unless `--minimal`, `Dashboard  Running`; no line
   says `Setup mode` or `Restart required`.
2. **Public models.** `curl -s http://127.0.0.1:8899/v1/models` lists
   `"id":"vllm-sr/auto"`.
3. **The decision model.**

   ```bash
   curl -s "$ROUTER_ORIGIN/api/v1/inventory/model-runtime" | python3 -c '
   import json, sys
   ready = {d["name"]: d["ready"] for d in json.load(sys.stdin)["deployments"]}
   print(ready)'
   ```

   Pass: `{'primary': True}`. An empty `{}` means no rule asks the decision
   model (you left out the `domains` rule).
4. **Routing preview**, which runs the signals and decisions without asking
   the backend:

   ```bash
   for prompt in 'In one sentence, why does a Python function return None?' \
       'Explain how a hash table handles collisions.' 'What is the capital of France?'; do
     vllm-sr route preview --model vllm-sr/auto --prompt "$prompt" --json | python3 -c '
   import json, sys
   d = json.load(sys.stdin)
   print(d["routing_decision"], d["selected_model"])'
   done
   ```

   Pass: `code-route`, `cs-route` and `default-route`, each with `MODEL`. Add
   `--trace` to see each condition; a decision-model rule that got no answer
   is listed under `signal_errors`.
5. **A routed answer:**

   ```bash
   curl -s -D - -o answer.json http://127.0.0.1:8899/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model": "vllm-sr/auto", "max_tokens": 64,
          "messages": [{"role": "user", "content": "In one sentence, why does a Python function return None?"}]}' \
     | grep -iE '^(HTTP|x-vsr-|server)'
   python3 -c 'import json; c = json.load(open("answer.json"))["choices"][0]; print(c["finish_reason"], repr(c["message"]["content"][:200]))'
   ```

   Pass: `HTTP/1.1 200 OK`, `x-vsr-selected-decision: code-route`,
   `x-vsr-selected-model: MODEL`, `x-vsr-selected-recipe: default`,
   `x-vsr-response-path: upstream`, `x-vsr-schema-version: 3`, and non-empty
   content. `server: envoy` appears only with `--gateway extproc`.
6. **Probe**, the CLI's own end-to-end assertion:

   ```bash
   vllm-sr route probe --config config.yaml --model vllm-sr/auto \
     --prompt 'In one sentence, why does a Python function return None?' \
     --max-completion-tokens 256 --expect-decision code-route --expect-selected-model MODEL
   ```

   Pass: exit code 0, `"passed": true` and `"status": 200` under `response`.
   It exits 2 when an assertion fails, including an answer cut off at the
   token cap (`finish_reason: length`), so keep a prompt that asks for a short
   answer and a cap with room to spare.

Then [hand off](https://vllm-sr.ai/install/agent/vllm-sr/SKILL.md#hand-off).

## GPU platforms

`--platform rocm` selects the `vllm-sr-rocm` image (about 20 GB on disk) and
passes `/dev/kfd` and `/dev/dri` into the Router container. `--platform cuda`
selects `vllm-sr-cuda` and gives the container every GPU; it needs the NVIDIA
Container Toolkit (validated on H200, A100 and Tesla T4). On macOS the Docker
target runs on the CPU only.

- The Router container sees every GPU. To keep some for the user's own model
  server, set `VLLM_SR_AMD_ROUTER_VISIBLE_DEVICES` (it becomes
  `ROCR_VISIBLE_DEVICES`, for example `1`) before `serve`.
- A model the Router runs is a `global.model_catalog.deployments` entry with
  `provider: model_runtime` and `device: rocm:N`, `cuda:N`, `cpu` or `auto`
  (the first validated GPU with enough free memory, else the CPU). A named GPU
  must exist, or the model fails to load with the reason. See
  [Choose a model](https://vllm-sr.ai/docs/model-runtime/choose-a-model).
- Only the deployments that a rule, a feature or a native grant resolves to
  start. `GET $ROUTER_ORIGIN/api/v1/inventory/model-runtime` shows each one's
  `state`, `ready`, `device` and replicas; a running process alone does not
  prove readiness.
- `--platform auto` inspects the actual container host. If both CUDA and ROCm
  are available, select one explicitly.

On a GPU host, also check one model on the GPU in an isolated Engine stack, so
that the Router stack and its grants stay unchanged:

```bash
(
  probe_dir=$(mktemp -d)
  cd "$probe_dir" || exit 1
  export VLLM_SR_STACK_NAME=vllm-sr-gpu-probe VLLM_SR_PORT_OFFSET=200
  export VLLM_SR_STATE_ROOT_DIR="$probe_dir"
  vllm-sr serve vllm-sr/Decision-2.0-Kai-0.6B -e --platform rocm --device-ids 0
  vllm-sr instance models
  curl -fsS http://127.0.0.1:9099/v1/systemone -H 'content-type: application/json' -d '{
    "model": "vllm-sr/Decision-2.0-Kai-0.6B",
    "state": {"request": "Write a Python function that merges two sorted lists."},
    "questions": {"task": {"type": "choice", "instructions": "What kind of work is this?",
      "choices": [{"key": "code", "description": "Writing or fixing code"},
                  {"key": "chat", "description": "Anything else"}]}}}'
  vllm-sr stop
)
```

Pass: `serve` prints `✓ Ready: http://localhost:9099/v1/systemone`, the
inventory reports a ready replica on the chosen GPU, and the answer has
`"choice": "code"`. Use `--platform cuda` on NVIDIA; choose an available host
GPU instead of assuming GPU 0 is free. Preserve the source directory, model
cache, images and volumes after stopping.

## Engine mode and replicas

`vllm-sr serve MODEL --engine` (or `-e`) serves a decision model without
recipe routing: only the Router container starts (the frontend, the native
System One API and the model runtime), and `--dashboard`, `--observability` or
`--no-minimal` add the rest. Without `-e`, every invocation starts Router mode.

The frontend publishes `POST /v1/systemone` and `GET /v1/systemone/models`
according to `listeners[].systemone.models` and the listener's API keys. A body
is a standard request (`state` and `questions`) or a batch (`states`, a map of
named standard requests), never both; see
[the System One API](https://vllm-sr.ai/docs/system-one/api). The Router and its model
runtime speak runtime contract 4.0.0 and must come from one release.

MODEL replaces only the default decision deployment's artifact; without it,
`serve` keeps the configured one, and a new configuration uses vela3-fast.
`--max-num-seqs`, `--max-num-batched-tokens`, `--batch-window-ms`, `--dtype`
and `--enable-shared-context` set the deployment's engine arguments, and
`--revision` pins a branch, tag or commit once at startup.

`-dp N` (`--data-parallel-size N`) writes N replicas. A new placement uses the
first N available GPUs; `--device-ids 0` puts all N on one host GPU, and
`--device-ids 0,1` one per GPU. See the
[model runtime Quickstart](https://vllm-sr.ai/docs/model-runtime/quickstart).

## Envoy in front: `--gateway extproc`

Standalone serves the same routing core without Envoy. Use `extproc` for
Envoy's rate limiting, mTLS, JWT or OIDC and advanced route matching, for an
Envoy-based gateway the user runs, or for policies that read the caller's
identity when no listener trusts identity headers
([What needs extproc](https://vllm-sr.ai/docs/installation/gateway-modes#what-needs-extproc)).

On Docker, switching is a restart on the active configuration:
`vllm-sr serve --config config.yaml --gateway extproc`, and plain
`vllm-sr serve --config config.yaml` to come back. With Envoy, responses carry
`server: envoy`, `vllm-sr logs envoy` works, and `serve` prints
`✓ vLLM Semantic Router is running (extproc gateway)`. A custom gateway routes
on the `x-vsr-route-model` header (formerly `x-selected-model`).

An authenticator in front (Envoy `ext_authz` or the user's gateway) is also
what lets an upstream act as each user: it writes the user's credential to
the header `global.services.authz.delegated_credential` names, and only the
upstreams marked `auth: {credential: delegated}` receive it
([Upstream credentials](https://vllm-sr.ai/install/agent/vllm-sr/references/configuration-loop.md#upstream-credentials)).

## Kubernetes with Helm

Use a complete configuration (step 3); setup mode isn't available on
Kubernetes. Model endpoints must be reachable from the pods: a Service DNS
name or an address on the cluster's network, not `host.docker.internal`. Set
the listener's `address` to `0.0.0.0`: the Service, not the host, publishes it.

You need `kubectl` with a context for the target cluster, and `helm` 3. A pip
or curl install carries no chart: `serve --target kubernetes` installs
`deploy/helm/semantic-router` when the current directory has one, and
otherwise the published chart that matches the CLI; `--chart-dir` names
another. `status`, `logs`, `stop` and `dashboard` need no chart.

```bash
vllm-sr config validate --config config.yaml
kubectl get storageclass
vllm-sr serve --target kubernetes --config config.yaml --namespace vllm-sr
vllm-sr status --target kubernetes --namespace vllm-sr
kubectl port-forward -n vllm-sr svc/semantic-router 8899:8899
```

Add `--context` to name a kubectl context, `--platform rocm` or `cuda` for the
GPU image and resources (`amd.com/gpu` or `nvidia.com/gpu`), `--gateway
extproc` for an Envoy-based gateway in the cluster, and `--minimal` to leave
out the observability stack. The Router keeps its models on a persistent
volume of storage class `standard`; `--profile dev` asks for the cluster's
default class instead (and enables the Dashboard). The chart's default memory
limit, 12Gi, covers the CPU decision model; a node needs that much free.
`serve` runs `helm upgrade --install semantic-router --wait --atomic`, so a pod
that doesn't become ready rolls the release back.

With the port forward running, steps 5.2 to 5.6 apply unchanged.
`vllm-sr stop --target kubernetes --namespace vllm-sr` uninstalls the release.
Sensitive environment values go into a Secret the CLI manages, never into
Helm values. Later Helm upgrades keep a document the Dashboard or Router API
saved; see
[Configuration Workflows](https://vllm-sr.ai/docs/installation/configuration-workflows#helm).

## Kubernetes with the operator

The operator reconciles a `SemanticRouter` resource into a Router Deployment,
Service and canonical configuration. Install it once per cluster (it needs
permission to install CRDs and cluster RBAC); ask before you do:

```bash
git clone https://github.com/vllm-project/semantic-router.git
cd semantic-router/deploy/operator
make install
make deploy IMG=ghcr.io/vllm-project/semantic-router/operator:latest
kubectl get pods -n semantic-router-operator-system
```

Pin a released image tag rather than `latest` for a controlled environment.
A first Router binds an existing model Service and routes with v0.5 recipes
under `spec.config.recipes`:

```yaml
apiVersion: vllm.ai/v1alpha1
kind: SemanticRouter
metadata:
  name: my-router
  namespace: default
spec:
  replicas: 1
  vllmEndpoints:
    - name: local-backend
      model: local/model
      backend:
        type: service
        service:
          name: model-server
          port: 8000
  config:
    models:
      - name: local/model
        capabilities: [chat]
    recipes:
      - name: default
        decisions:
          - name: default-route
            description: Route every request to the discovered model.
            priority: 1
            rules:
              operator: AND
              conditions: []
            modelRefs:
              - model: local/model
```

```bash
kubectl apply -f my-router.yaml
kubectl wait --for=condition=Available deployment -l app.kubernetes.io/instance=my-router --timeout=900s
kubectl port-forward -n default service/my-router 8801:8801
```

Pass: the resource's `status.observedGeneration` equals
`metadata.generation`, the deployment is Available, and a completion through
port 8801 (the operator's standalone listener, not 8899) returns 200 with the
`x-vsr-*` headers. `spec.config.routing` is refused; `spec.config.entrypoints`,
`spec.config.plugins` and `spec.config.models` take the canonical entrypoints,
`global.plugins` and model facts. Reference provider keys from `spec.env`
with `secretKeyRef`. See the
[operator guide](https://vllm-sr.ai/docs/installation/k8s/operator).

## Dashboard and access

The Dashboard and every other port publish on 127.0.0.1. The listener's own
`address` decides the inference port's publication. To publish the Dashboard
beyond loopback, set `VLLM_SR_DASHBOARD_HOST_BIND=0.0.0.0` together with
`DASHBOARD_ADMIN_EMAIL` and `DASHBOARD_ADMIN_PASSWORD` (or an explicit
`DASHBOARD_ALLOW_OPEN_BOOTSTRAP=true`); `serve` refuses otherwise. On loopback,
the first visitor creates the administrator unless those variables were set
before `serve`. `VLLM_SR_OBSERVABILITY_HOST_BIND=0.0.0.0` publishes Jaeger,
Prometheus and Grafana beyond loopback; Grafana's login is `admin/admin` until
someone changes it. Inspect the actual bindings with `docker ps` before opening
access; a tunnel doesn't close a port that is already public.

## Stack identity

One host runs one stack per `VLLM_SR_STACK_NAME`. For a second stack, set a
different name and a `VLLM_SR_PORT_OFFSET`, which moves every host port by the
same amount, and use both values for every later command of that stack. Its
containers are named `NAME-vllm-sr-router-container` and so on. An inherited
`VLLM_SR_STATE_ROOT_DIR` can load another stack's active configuration: point
it at the directory that contains `.vllm-sr`, usually the config directory, or
leave it unset.

## Run a specific image

`vllm-sr serve` pulls the images published for the CLI's own build. To run
other images, such as a mirror, a pinned digest or a development build already
on the host, export the image for every command of the stack, and don't pull
it:

```bash
export VLLM_SR_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr:TAG
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml --platform cpu --image-pull-policy ifnotpresent
```

`serve` checks a new configuration with the image it is about to start
(`--image`, else `VLLM_SR_ROUTER_IMAGE` or `VLLM_SR_IMAGE`) before it stops
the running stack. `config validate` uses `--image` or `--router-binary`
when given, else the same image `serve` would run. `config migrate` has no
`--image` option: it uses `VLLM_SR_ROUTER_BINARY`, else the variables above,
which is why the export covers every command. Neither validate nor migrate
pulls an image or asks a running stack.

An official `vllm-sr` image name also selects the matching `dashboard` image of
the same tag; `VLLM_SR_DASHBOARD_IMAGE` names another. With `--platform rocm`
or `cuda`, use the `vllm-sr-rocm` or `vllm-sr-cuda` image.

## Add or replace a backend

1. Check the backend's protocol, model name and capabilities with a bounded
   direct request, keeping credentials in environment variables.
2. Add a `providers.models[]` entry with its `backend_refs` and the facts
   routing reads (`capabilities`, `modalities`, context window), then name it in
   a decision's `modelRefs`; a provider entry alone isn't routable.
3. Plan and apply the change ([Configuration](https://vllm-sr.ai/install/agent/vllm-sr/references/configuration-loop.md)), then
   preview and probe the affected routes. A new listener needs a restart.

Size the Router's own models apart from the generation backends: resident
weights, memory for the intended context and concurrency, and disk.
