# Maintain vLLM Semantic Router

Upgrades, the v0.5 configuration migration, configuration history, backups,
secrets, logs and health, and removal. Run every command from the stack's
config directory with its `VLLM_SR_STACK_NAME` and `VLLM_SR_PORT_OFFSET`, and
set `ROUTER_ORIGIN` to its management origin. Ask before any step that stops a
stack the user runs or deletes data.

## Record and back up first

```bash
vllm-sr --version
vllm-sr status
vllm-sr config versions
tar czf "vllm-sr-backup-$(date +%Y%m%d%H%M%S).tgz" config.yaml .vllm-sr
chmod 600 vllm-sr-backup-*.tgz
```

Pass: the archive exists and `tar tzf` lists `config.yaml` and
`.vllm-sr/runtime-config.yaml` (`runtime-config.NAME.yaml` for a stack named
`NAME`). `.vllm-sr/` holds the active document, the configuration history, the
Dashboard's data, the recipe store and the stack's generated credentials, so
keep the archive as private as `.vllm-sr/` itself and never print its
secrets. `models/` holds downloaded models, which the stack downloads again.

Redis and Postgres keep their data in named volumes: `vllm-sr-redis-data` and
`vllm-sr-postgres-data` for the default stack, `NAME-vllm-sr-redis-data` and
`NAME-vllm-sr-postgres-data` for a stack named `NAME`. Copy one while the stack
is stopped:

```bash
docker run --rm -v vllm-sr-redis-data:/data:ro -v "$PWD":/backup alpine \
  tar czf /backup/vllm-sr-redis-data.tgz -C /data .
```

To restore, run the same container with the volume writable and
`tar xzf /backup/vllm-sr-redis-data.tgz -C /data`, before the next `serve`.
Milvus, Qdrant and Valkey services you run have their own backups.

## Upgrade within v0.5

For a stack whose `config.yaml` already says `version: v0.5`:

```bash
curl -fsSL https://vllm-sr.ai/install.sh | \
  bash -s -- --channel dev --mode cli --runtime skip --no-launch
vllm-sr --version
vllm-sr serve --config config.yaml
```

Use the channel the stack was installed from. `serve` pulls the images of the
new CLI (`--image-pull-policy always` by default) and restarts the stack on its
active configuration. Pass: a newer version prints, `serve` exits 0 after
`✓ vLLM Semantic Router is running`, and the stack's own checks pass again
(status, preview and probe, as in
[Deployment](https://vllm-sr.ai/install/agent/vllm-sr/references/deployment-loop.md#5-check-the-stack)).

## Migrate a configuration to v0.5

v0.5 is one breaking release. A configuration that says `version: v0.3`, or
uses any removed form, fails to load with one line that names the form and
what it becomes:

```yaml removed
version: v0.3
routing:
  signals:
    keywords:
      - name: urgent
        operator: OR
        keywords: [urgent, asap]
```

```text
version: invalid_value: config schema v0.3 is removed; run `vllm-sr config migrate` (version: v0.5)
```

`vllm-sr config validate` prints it after `Error: Configuration parsing
failed:` and exits 1. `vllm-sr config migrate` rewrites every such form, from
a configuration of any earlier release in one step: the flat layout before
config v0.3 (no `version`, top-level `vllm_endpoints`, `model_config` or
`classifier`) included. There is no detour through an older CLI.

Every rewrite lives in the Router, so migrate asks one, never over the
network: the binary `VLLM_SR_ROUTER_BINARY` names, else the image
`VLLM_SR_ROUTER_IMAGE` or `VLLM_SR_IMAGE` names, else this CLI's own release
image. That image must already be on the host: migrate never pulls it and
never asks a running stack's Router, so it doesn't matter whether the old
stack still runs. Migrate has no `--image` option; to migrate with a pinned
image, export `VLLM_SR_IMAGE`
([Run a specific image](https://vllm-sr.ai/install/agent/vllm-sr/references/deployment-loop.md#run-a-specific-image)).

Migrate writes only a document the Router accepts for the gateway it will
serve in: standalone, unless you pass `--gateway extproc` as `config validate`
takes it. Earlier releases served behind Envoy, and a configuration whose
policies read the caller's identity (authz signals and providers, per-user
rate limits, `debug: authenticated`) needs that again: standalone refuses it
with `… needs an identity source in standalone mode; …` and the hint
``Re-run with `--gateway extproc` when Envoy or another authenticating gateway
sits in front of the Router, …``. Migrate, validate and serve such a file
with `--gateway extproc`. Behind the
user's own authenticating proxy, `listeners[].identity.trust_headers: true`
serves it standalone instead; ask first.

To upgrade a stack of an earlier release:

1. Back up (above). The old stack keeps serving until step 5 restarts it;
   agree on that restart with the user.
2. Install the new CLI (step 2 of [Deployment](https://vllm-sr.ai/install/agent/vllm-sr/references/deployment-loop.md#2-install-the-cli)).
3. Migrate, keeping its output:

   ```bash
   vllm-sr config migrate --config config.yaml > migrate.log 2>&1; echo "migrate exit=$?"
   ```

   For a file that reads the caller's identity (above), run
   `vllm-sr config migrate --config config.yaml --gateway extproc` instead,
   and pass the same `--gateway extproc` to validate and serve in steps 4
   and 5. When this CLI's Router image isn't on the host yet, migrate exits
   1 with `No Router could migrate the configuration:` and the exact command
   that fetches it, such as `` `docker pull IMAGE` fetches the one `vllm-sr
   serve` runs on this host ``, or, for a CLI run from a source checkout,
   ``build … with `make -C REPO vllm-sr-build` ``. Run that command, then
   migrate again.

   Pass: exit 0 after `✓ Configuration migrated`, `Output` names
   `config.migrated.yaml`, and `migrate.log` lists the `Changes to review`
   and the `Warning:` lines, which say what an operator must still do for
   the file to work as before. **Read every one before you serve.** They
   cover re-fitted thresholds, dropped weights or example banks, conditions
   migrate couldn't split, `safety` hazard categories rewritten as harm
   categories, dropped `classify`, `http` and `mom_registry` references (the
   decision model answers those signals),
   `global.model_catalog.modules.prompt_compression` moved to
   `compress.signals`, and the v0.4 forms in
   [the table below](https://vllm-sr.ai/install/agent/vllm-sr/references/maintenance.md#what-migrate-does-with-v04-forms). Restore each
   dropped form the user's traffic needs in `config.migrated.yaml`, with the
   replacement its note names (ask when it needs something only the user
   knows, such as an endpoint), and report every change, drop and warning.
4. Validate and swap the files:

   ```bash
   vllm-sr config validate --config config.migrated.yaml
   mv config.yaml config.v0.4.yaml
   mv config.migrated.yaml config.yaml
   ```

   Pass: exit 0 after `✓ Configuration is valid` and `Checked by the Router
   in IMAGE as well as the CLI.`, where IMAGE is the one migrate used.
5. Serve the migrated document in place of the old active one:

   ```bash
   vllm-sr serve --config config.yaml --replace-active-config
   ```

   Before it stops the old stack, `serve` checks the new document with the
   Router image it is about to start, pulled as `--image-pull-policy` says. A
   refusal (`Error: Invalid runtime config: the Router in IMAGE refuses it:
   …`) or a failed pull exits 1 and leaves the old stack running.
   `--replace-active-config` replaces `.vllm-sr/runtime-config.yaml`, which
   still holds the old schema; Dashboard edits made on the old version are
   only in the backup, so ask first when the old active document differs from
   the old `config.yaml`.
6. Run the stack's checks. A gateway or client that read the old headers needs
   the new ones: `x-vsr-route-model` for a custom Envoy route,
   `x-vsr-response-path` for cache hits, `x-vsr-trace` members for signals.

Failure branches:

- `Error: Output file already exists: config.migrated.yaml`: an earlier run
  wrote it. Rerun with `--force` to regenerate it; the output is the same for
  the same input.
- `No Router could migrate the configuration: the Router image IMAGE, which
  VLLM_SR_IMAGE names, is not present locally; …`: the image you named isn't
  on this host. Run the command the message names, or name an image that is
  (`docker image ls 'ghcr.io/vllm-project/semantic-router/*'`).
- A refusal that ends with ``. `latest` is a moving tag: this local copy was
  built on DATE and may be older than this CLI; …``: a development CLI runs
  the moving `latest` images, and the copy on this host predates the CLI.
  Run the pull or the `make -C REPO vllm-sr-build` the message names, or name
  the image you mean with `VLLM_SR_IMAGE`, then rerun.
- `… needs an identity source in standalone mode; …`: the file reads the
  caller's identity. Migrate it again with `--gateway extproc` (above), and
  serve it with `vllm-sr serve --config config.yaml --replace-active-config --gateway extproc`.
- A form with no rewrite at all, the `image_gen` plugin or an audio or
  image-anchor `embeddings` rule, fails the migration and says after
  `it has no mechanical rewrite:` what to write instead. Edit a copy of the
  file as the message says (deleting a form the user doesn't use is an edit
  to report), and rerun.
- A Kubernetes operator resource is not a file: rewrite `spec.config.routing`
  as `spec.config.recipes` by hand before upgrading the CRDs.

Migrating a file that is already v0.5 rewrites nothing. The
[migration guide](https://vllm-sr.ai/docs/installation/migrate-to-v0-5) lists every removed form,
its load error and its rewrite.

### What migrate does with v0.4 forms

The stock v0.4.0 configuration migrates in one run. Migrate rewrites what
maps onto v0.5 and drops what v0.5 can't execute or no longer reads, with a
note on each:

| v0.4 form | What migrate writes | What to restore |
| --- | --- | --- |
| `global.integrations.looper.<remom, fusion or flow>.model_names` | An entrypoint of those names that serves a new recipe, `default-remom` for example: the default recipe with only its decisions of that algorithm. A request none of them matches now goes to the default model instead of failing. Names that served no decision are dropped. | Nothing |
| `global.router.auto_model_names` beside the file's own default entrypoint | An entrypoint of the auto names (`vllm-sr/auto`, `auto`, …) for the default recipe | Nothing |
| `kb` rules, `kb` conditions and `kb_metric` score inputs | Nothing: no v0.5 signal reads a knowledge base, so they never match. A condition on one is false, which can drop the decision that needs it. | The `systemone` question over the knowledge base's labels that its note gives |
| A `structure` rule on `text_bytes` | A `keywords` regex rule whose match count is the request's UTF-8 byte length, with the same `count` predicate | With `compress.signals` on, add `keyword` to its `skip_signals` to count the original text |
| An image route on a model with `api_format: openai` | The model's `api_format: images` (`responses` when it also answers in text) | Check that its backend serves `POST /v1/images/generations` |
| A route that answers with text and an image over two models | Nothing: no decision of this release executes it, so its requests go to the other decisions | Models on `api_format: responses` that answer in text and generate images, with `output: {kind: [text, image], action: generate}` |
| `recipes[].data_policy` | The recipe's `router_replay` plugin: `enabled: false` for `replay: false`, `prompt_capture: verified` for `replay_personal_data: false` | Nothing |
| An enabled RAG source on `backend: mcp` | Nothing: the Router has no MCP tool invoker, so those requests retrieve nothing | Serve the retrieval over HTTP, and add the `{type: rag, backend: external_api, …}` source its note gives |
| A deployment's `operating_point` | Nothing: nothing read it, and the model runtime applies the one its package declares | Nothing |

## Versions and rollback

Every activation takes the next version number, and the Router keeps the last
10 documents in `.vllm-sr/config-backups`:

```bash
vllm-sr config versions
vllm-sr config rollback 3
```

`config rollback` takes a version number or a backup's timestamp and activates
the recorded document as a new version. Pass: it exits 0, and
`config versions` lists the new version first. To return to an older CLI
release after an upgrade, reinstall it (`--pip-spec 'vllm-sr==VERSION'`),
restore the backup and run `vllm-sr serve --config config.yaml
--replace-active-config`. On Helm, `helm rollback semantic-router -n NAMESPACE`
restores the previous release.

## Secrets

- Provider keys: name the variable (`auth: {api_key_env: VAR}` on a backend)
  and export it before `serve`; a recipe that reads host variables needs
  `vllm-sr serve --recipe-env VAR`. On Kubernetes, the CLI moves sensitive
  values into a Secret, and the operator reads them from `spec.env`
  `secretKeyRef`s. An upstream that must act as each user takes
  `auth: {credential: delegated}` and no key
  ([Upstream credentials](https://vllm-sr.ai/install/agent/vllm-sr/references/configuration-loop.md#upstream-credentials)).
- Storage credentials: each stack generates its own Redis and Postgres
  credentials in `.vllm-sr/storage-secrets/`. Rotate them with
  `vllm-sr storage rotate`, then rerun the same `serve` command; new
  connections fail until it runs.
- The Dashboard's management credential lives in
  `.vllm-sr/management-credential/`. Delete its file and rerun `serve` to
  rotate it, or provide `VLLM_SR_DASHBOARD_RECIPE_TOKEN` yourself.
- The management API takes a bearer token from the variable `--token-env`
  names (default `VSR_MGMT_TOKEN`).

## Logs and health

| Check | Command | Pass |
| --- | --- | --- |
| Containers | `vllm-sr status` | `State  Running`; no `Setup mode` or `Restart required` |
| Readiness | `curl -s "$ROUTER_ORIGIN/ready"` | `"ready":true` |
| Status conditions | `curl -s "$ROUTER_ORIGIN/api/v1/status"` | `Live`, `StartupComplete`, `ActiveConfigAvailable` and `ConfigConverged` are `"True"`; `config.activation_status` is `active`. `Ready` stays `"Unknown"` with `RequiredDependenciesUnobserved` on a healthy stack: this report doesn't observe dependencies, so read readiness from `/ready` and the inventory |
| Decision model | `curl -s "$ROUTER_ORIGIN/api/v1/inventory/model-runtime"` | each deployment `"ready":true`, `"restarts"` steady |
| Last rejection | `curl -s "$ROUTER_ORIGIN/api/v1/config/hash"` | no `last_rejection`, or one you already handled |
| Metrics | `curl -s http://127.0.0.1:9190/metrics` | `llm_config_active_version` is the version you expect |
| Logs | `vllm-sr logs router`, `vllm-sr logs dashboard`, `vllm-sr logs envoy` | the last 200 lines; add `-f` to follow |

On Kubernetes, `vllm-sr status --target kubernetes --namespace NAMESPACE` and
`vllm-sr logs router --target kubernetes --namespace NAMESPACE`. A stack that
should survive host reboots starts with `vllm-sr serve --config config.yaml
--restart-policy unless-stopped`.

## Stop and remove

- **Stop:** `vllm-sr stop` removes the containers and networks and prints
  `✓ vLLM Semantic Router stopped`; `vllm-sr status` then says
  `Not running`. It keeps `config.yaml`, `.vllm-sr/`, `models/`, the data
  volumes and the images.
- **Remove**, only when the user asks, because it deletes their data: after
  `vllm-sr stop`, delete the stack's volumes, each named after its container
  (`docker volume ls --format '{{.Name}}' | grep -E
  '^vllm-sr-(redis|postgres|grafana|jaeger)-data$'`, with the `NAME-` prefix
  for a stack named `NAME`), the config
  directory's `.vllm-sr/` and `models/`, the images (`docker image ls --filter
  'reference=ghcr.io/vllm-project/semantic-router/*'`), the Engine-mode state
  `~/.local/state/vllm-sr`, the sr-bench harnesses in `~/.cache/vllm-sr`, and
  the CLI (`~/.local/share/vllm-sr`, `~/.local/bin/vllm-sr`, and the
  completion line in the shell's rc file). Other stacks' volumes carry their
  own `NAME-` prefix; leave them.
- **Kubernetes:** `vllm-sr stop --target kubernetes --namespace NAMESPACE`
  uninstalls the Helm release, including its volume claims (the downloaded
  models and a single replica's configuration history), so ask first.
