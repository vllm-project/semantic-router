# Semantic Router Dashboard

The Dashboard is the authenticated control and observability UI for a Semantic
Router deployment. It combines a React frontend with a Go backend that serves
the SPA, stores dashboard state, and proxies Router, Envoy, Grafana,
Prometheus, and Jaeger endpoints.

Use it to:

- complete first-run model and recipe setup;
- inspect, edit, validate, deploy, and roll back Router configuration;
- import and activate Recipe packages;
- test routes in the Playground and inspect the selected path;
- view topology, logs, evaluations, and monitoring tools;
- manage security policies, ML selection workflows, MCP tools, and optional
  OpenClaw workers when those features are enabled.

The Dashboard is a control plane, not an inference proxy. Applications should
send inference requests to Envoy.

## Local development

Start the frontend and backend in separate terminals from the repository root:

```bash
make dashboard-dev-frontend
```

```bash
ROUTER_CONFIG_PATH="$PWD/config/config.yaml" \
TARGET_ROUTER_API_URL=http://127.0.0.1:8080 \
TARGET_ENVOY_URL=http://127.0.0.1:8899 \
make dashboard-dev-backend
```

Open `http://127.0.0.1:3001`. Vite proxies backend requests to port `8700`.
The Router and Envoy must be running for live config, status, and Playground
operations.

For the complete local stack, use the CLI instead:

```bash
vllm-sr serve --config config/config.yaml
vllm-sr dashboard
```

The current installation and first-run workflow is documented in
[`website/docs/installation/installation.md`](../website/docs/installation/installation.md).

## Build and test

```bash
make dashboard-build
make dashboard-check
make dashboard-test-e2e-evaluation
make dashboard-test-backend
```

The required `Dashboard` CI workflow runs `dashboard-check`, then the sr-bench
browser acceptance target shown above. Both gates are available locally; run
`dashboard-check` before every Dashboard change and the browser acceptance gate
when changing sr-bench UI or workflows. `dashboard-check` runs, in order:

| Step | What it covers |
| --- | --- |
| `dashboard-lint` | ESLint on the frontend, golangci-lint on the backend |
| `dashboard-type-check` | TypeScript type checking (frontend + Knowledge Map) |
| `dashboard-test-frontend` | Frontend unit tests |
| `dashboard-test-backend` | Go test inventory and JSON test evidence on `dashboard/backend`, including authentication, ownership forwarding and the sr-bench service proxy |
| `dashboard-go-mod-tidy` | Verifies `go.mod` / `go.sum` are tidy |

The dashboard backend is a **separate Go module**, so `go test ./...` from the
repository root does not cover it. Use `make dashboard-test-backend`, or run
`go test ./...` from `dashboard/backend` directly.

Some backend tests shell out to the `vllm-sr` CLI (for example to regenerate Envoy
config), so they need its Python dependencies importable:

```bash
pip install -e src/vllm-sr
```

Without it, those tests fail with `ModuleNotFoundError`. CI installs the same package.

`dashboard-check` runs plain `go test`. The race detector roughly doubles the
runtime, which is a poor trade on every PR, so it is deliberately **not** in the
always-on gate. Run it locally before pushing concurrency-sensitive work — anything
touching shared state, goroutines, caches, or resolvers:

```bash
cd dashboard/backend && go test ./... -race
```

## Runtime configuration

The backend accepts matching command-line flags for these environment
variables. Defaults are defined in
[`backend/config/config.go`](backend/config/config.go).

| Variable | Purpose |
| --- | --- |
| `DASHBOARD_PORT` | Backend listen port; default `8700`. |
| `DASHBOARD_STATIC_DIR` | Built frontend assets. |
| `ROUTER_CONFIG_PATH` | Canonical Router YAML read or updated by config APIs. |
| `DASHBOARD_CONFIG_DIR` | Directory for config versions and related state. |
| `TARGET_ROUTER_API_URL` | Router management API; default `http://localhost:8080`. |
| `TARGET_ROUTER_METRICS_URL` | Router Prometheus endpoint. |
| `TARGET_ENVOY_URL` | Inference endpoint used by Playground and route probes. |
| `TARGET_GRAFANA_URL` | Optional Grafana base URL. |
| `TARGET_PROMETHEUS_URL` | Optional Prometheus base URL. |
| `TARGET_JAEGER_URL` | Optional Jaeger base URL. |

Feature controls:

| Variable | Purpose |
| --- | --- |
| `DASHBOARD_READONLY` | Hard-disable all config mutation. |
| `DASHBOARD_RUNTIME_CONFIG_WRITABLE` | Allow mutation of the mounted runtime config surface. |
| `DASHBOARD_RECIPE_STORE_WRITABLE` | Allow Recipe package import. |
| `DASHBOARD_SETUP_MODE` | Enable the trusted first-run setup flow. |
| `SR_BENCH_URL` | Server-owned sr-bench service origin; default `http://127.0.0.1:8090`. |
| `SR_BENCH_TOKEN_ENV` | Environment variable containing the service token; default `SR_BENCH_TOKEN`. The browser never receives this token. |
| `ML_PIPELINE_ENABLED` | Enable benchmark, training, and config-generation jobs. |
| `ML_TRAINING_DIR` | Training script directory for subprocess mode. |
| `ML_SERVICE_URL` | Use an ML service instead of local subprocesses; co-located sidecars use `http://127.0.0.1:8686`. |
| `MCP_ENABLED` | Enable MCP server and tool management. |
| `OPENCLAW_ENABLED` | Enable OpenClaw provisioning and room workflows. |

OpenClaw provisioning accepts optional `skills` entries as exact IDs from the
server's skills catalog (`GET /api/openclaw/skills`). IDs use lowercase ASCII
letters or digits, with single hyphens or underscores separating groups. Paths,
case or whitespace aliases, and unknown IDs return HTTP 400 before provisioning
starts, including for asynchronous requests. Malformed catalog JSON returns
HTTP 500 when skills are selected. Omitting skills or selecting an empty list
still provisions without skills. Administrators can supply a catalog with
`OPENCLAW_SKILLS_PATH`.

## sr-bench evaluation

The Evaluation page is sr-bench 1.0. It uses the same durable Python service as
`vllm-sr benchmark`; the Go backend authenticates each request and forwards the
user identity. Closing the browser or restarting the Dashboard does not cancel
a run. The benchmark service owns execution, persisted results and cancellation.

`vllm-sr serve` starts an independent core benchmark worker with the Dashboard
image. Its store is `<state-root>/.sr-bench/<stack>/store`; its private service
token is adjacent to that store, outside the Router and Dashboard mounts. The
worker publishes a loopback port, `8090 + port offset`, and receives neither a
Docker socket nor GPU devices. Dashboard/config reloads reuse a matching running
worker. A stopped or changed worker requires explicit reconciliation; `vllm-sr
stop` stops it without deleting its evidence.

The core image does not include every upstream execution environment. For code
and interactive benchmarks, prepare a dedicated worker host with the required
pinned harnesses and sandbox dependencies. `SR_BENCH_URL` selects that external
worker and suppresses local worker creation. Use an origin reachable from the
Dashboard container and the same server-side token for both clients. In local
Dashboard development:

```bash
# Set SR_BENCH_TOKEN to the same private value in both server environments.
vllm-sr benchmark --store ./data/sr-bench serve
SR_BENCH_URL=http://127.0.0.1:8090 make dashboard-dev-backend
```

The standalone service binds loopback by default. Keep the token server-side and use the authenticated
Dashboard origin for browser access. Register operator-owned targets with
`vllm-sr benchmark --store ./data/sr-bench target register --file targets.json`.
Targets contain endpoint and model identities, four token prices, and credential
environment references. The Dashboard selects registered targets; it cannot
redirect their credentials to another endpoint.

Prepare versioned datasets with `vllm-sr benchmark dataset prepare` and select a
frozen dataset, profile, targets and limits in Evaluation. Review the plan before
starting. Live runs record capability and usage; preview runs record routing
diagnostics only. The page shows per-target and per-benchmark results, four
token buckets, latency, wall time, failures, routing distributions and case
evidence. Comparisons require completed live runs on the same frozen cases.
Unknown costs remain unknown.

Mount the sr-bench store as writable persistent storage. Its SQLite ledger and
run artifacts survive service restarts. Interrupted requests are preserved for
reconciliation and are never resent automatically. Dashboard-owned SQLite paths
remain `DASHBOARD_AUTH_DB_PATH`, `DASHBOARD_WORKFLOW_DB_PATH`, and
`DASHBOARD_CONFIG_PROJECTION_DB_PATH`. Historical Evaluation Plane files are not
imported or deleted by sr-bench.

## Authentication and write safety

Set a stable `DASHBOARD_JWT_SECRET` and provision the first administrator with
`DASHBOARD_ADMIN_EMAIL`, `DASHBOARD_ADMIN_PASSWORD`, and optionally
`DASHBOARD_ADMIN_NAME`. Public web-form bootstrap is disabled by default; only
set `DASHBOARD_ALLOW_OPEN_BOOTSTRAP=true` in a controlled first-run environment.

To keep local Docker or Podman sessions valid when recreating the stack:

1. Load the same `DASHBOARD_JWT_SECRET` into the host environment before every
   `vllm-sr serve` invocation. Use your existing secret store; do not generate a
   new value on each launch.
2. Keep the Dashboard authentication database on its persistent volume.
3. Start the stack normally. This variable configures Dashboard only; do not
   pass it through `--recipe-env`.

Without a stable key, each Dashboard restart requires users to log in again.
Rotating the key also ends existing sessions, but does not change stored
administrator accounts. If a temporary connection or server error interrupts
session verification, choose **Retry** to reconnect without signing in again.

Writes authenticated by the session cookie must carry an `X-CSRF-Token` header
and a matching `Origin`. The frontend does this on its own. Set
`DASHBOARD_ALLOWED_ORIGINS` to a comma-separated list when the browser's origin
differs from the backend's `Host`, as behind a reverse proxy or the Vite dev
proxy (`http://localhost:3001`). Unset, the origin check is advisory and the
CSRF token is the guarantee. `Authorization: Bearer` requests are exempt.

The same list governs the ClawRoom WebSocket handshake. CORS does not apply to
handshakes, so the origin check is the only cross-origin control there; a
split-origin frontend that is not listed can authenticate and write but cannot
open the room socket.

sr-bench APIs are intentionally stricter: they accept browser
requests only when `Origin` exactly matches the request scheme and `Host`.
TLS-terminating proxies must overwrite `X-Forwarded-Proto` with the external
scheme; arbitrary sibling origins never receive credentialed CORS headers.

Read-only mode and the two writable-surface flags are independent. A read-only
ConfigMap, GitOps-owned config, or read-only Recipe store should be reflected in
the matching flag so the UI does not offer operations the runtime cannot
persist.

Some local workflows can manage containers. Do not mount a container-runtime
socket unless users with Dashboard access are allowed to control that runtime.
See the [security hardening guide](../website/docs/installation/security-hardening.md)
for the deployment boundary.

## Session contract

Browsers authenticate with the `vsr_session` cookie the backend sets at login,
and with nothing else. It is `HttpOnly`, `SameSite=Lax`, and `Secure` behind
HTTPS, so page script cannot read it and the browser attaches it to same-origin
requests on its own — `fetch`, `EventSource`, `WebSocket`, and iframes alike.
The frontend does not store, copy, or forward it.

- **`?authToken=` is not accepted.** The backend used to read a session token
  from the query string, which put a live credential into reverse-proxy access
  logs, browser history, and the `Referer` header. Any saved link or automation
  still using it now receives `401`; move it to `Authorization: Bearer`.
  Token-shaped query parameters are redacted from the logs the dashboard writes,
  because old links keep arriving for a while.
- **Non-browser clients use `Authorization: Bearer`.** `POST /api/auth/login`
  returns the token in its response body for exactly this case. Bearer requests
  are exempt from the CSRF check described above, since a browser never attaches
  that header by itself.
- **`vsr_csrf` is readable by script on purpose.** The frontend reads it and
  copies the value into `X-CSRF-Token` on every unsafe request
  ([`frontend/src/utils/authFetch.ts`](frontend/src/utils/authFetch.ts)), which
  is why it is not `HttpOnly`. It is not a credential: it authenticates nothing
  on its own, and the server recomputes the expected value from the session id
  inside the session token rather than reading the cookie back, so planting one
  achieves nothing without the session cookie as well.
- **`SameSite=Lax` is deliberate.** `Strict` would withhold the cookie from
  top-level navigation into the dashboard, so following a link from chat or an
  alert would land on the login page despite a valid session. `Lax` still
  withholds it from cross-site subrequests and form posts, and the CSRF token
  covers what is left.

## Setup mode contract

Setup mode is the dashboard's first-run state. While it is active the UI forces the setup wizard and the **unauthenticated** first-admin bootstrap endpoint (`/api/auth/bootstrap/register`) is open, so what turns it on and off is a security boundary, not a cosmetic flag.

- **`setup.mode` in the router config file declares setup mode, and nothing else does.** It is read live from disk, behind an mtime+size cache, by one resolver (`dashboard/backend/setupmode`). Every surface derives from that single value: the bootstrap gate, `/api/setup/state`, `/api/settings`, and the setup write endpoints (`validate`, `activate`, `import-remote`).
- **What it enables:** the setup wizard, and creation of the first admin without logging in. Both end together.
- **It ends automatically when activation rewrites the config, with no restart.** Activation strips the `setup` block and the resolver's cache is invalidated in the same request, so the bootstrap endpoint closes at the moment setup finishes. Activation restarts the router and Envoy but deliberately not the dashboard, which is why the state must be read live rather than captured at startup.
- **`--setup-mode` / `DASHBOARD_SETUP_MODE` is deprecated and ignored.** It is still read, but only so that a value disagreeing with the config file can be reported: `/api/setup/state` returns a `reason`, and the backend logs one `WARNING` per change (not per request, since the endpoint is unauthenticated). A stale environment value can no longer open bootstrap on its own.
- **An unreadable or unparsable config resolves to "not in setup mode", deliberately.** Failing closed is the only safe posture for something gating unauthenticated admin creation; the resolver never falls back to the legacy flag on an error path. `/api/setup/state` answers `200` with a diagnostic `reason` (rather than a `500` the frontend silently coerced to "not in setup mode") so the condition is visible instead of silent. The reason never contains config file contents.
- **`--allow-open-bootstrap` is a separate, still-supported operator escape hatch.** It is unaffected by setup-mode resolution and has no config-file counterpart. Production should provision the admin via `DASHBOARD_ADMIN_*` rather than enabling it.

## Router contract access

The **System → Platform & Access → Router API Docs** entry opens the running
Router's Swagger UI through the authenticated Dashboard origin. Its companion
proxies are `/api/router/api/v1` and `/api/router/openapi.json`; they expose the
Router's `/api/v1` and `/openapi.json` responses through the existing read-only
management proxy and do not maintain another API definition. Agents should
query the Router endpoints directly and do not depend on the Dashboard.

## Architecture

```text
Browser
  -> React SPA (dashboard/frontend)
  -> Go API and reverse proxy (dashboard/backend)
       -> Router management API
       -> Envoy inference listener
       -> sr-bench service (durable execution and evaluation store)
       -> optional monitoring services
       -> local SQLite and config/Recipe storage
```

- [`frontend/src/app/`](frontend/src/app/) owns routing, authentication gates,
  and the application shell.
- [`frontend/src/pages/`](frontend/src/pages/) owns page orchestration.
- [`backend/router/`](backend/router/) registers public and authenticated API
  routes.
- [`backend/handlers/`](backend/handlers/) implements control-plane workflows.
- [`backend/recipe/`](backend/recipe/) validates and materializes Recipe
  packages.
- [`wizmap/`](wizmap/) builds the embedded knowledge-map view.

Keep detailed user workflows in the website and keep this README focused on
developing and operating the Dashboard itself.
