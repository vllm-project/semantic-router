# End-to-end test framework

The Go E2E runner builds the checked-out Router image, creates or reuses a Kind
cluster, deploys one named profile, runs that profile's registered test cases,
and writes `test-report.json` and `test-report.md` in the current directory.

A profile owns environment setup and a list of tests. A test case owns one
externally visible contract and can be reused by several profiles.

```text
e2e/cmd/e2e             command-line entry point
e2e/pkg/framework       cluster, profile lifecycle, execution, and reports
e2e/pkg/testcases       reusable test implementations and registry
e2e/profiles            deployment-specific profile implementations
e2e/config              focused Router configs for profiles and smoke tests
```

The scripts under [`testing/`](testing/) are older or specialized manual
utilities. New Kubernetes E2E coverage belongs in the Go runner unless the test
has a documented reason to remain a separate integration suite.

## Requirements

- Docker;
- Kind and `kubectl`;
- Helm;
- Go matching `e2e/go.mod`;
- enough local CPU, memory, and disk for the selected profile;
- credentials or accelerators declared by that profile.

The default run creates and deletes a Kind cluster named
`semantic-router-e2e`. Do not point `E2E_USE_EXISTING_CLUSTER=true` at a shared
or production cluster.

## Run a profile

From the repository root:

```bash
make e2e-test
```

The default profile is `envoy-ai-gateway`, which owns the broad Router contract.
Select another registered profile explicitly:

```bash
make e2e-test E2E_PROFILE=dashboard
make e2e-test E2E_PROFILE=routing-strategies
```

List the runner's current profile names and flags from the built binary:

```bash
make build-e2e
./bin/e2e -help
```

[`tools/agent/test-domain-registry.yaml`](../tools/agent/test-domain-registry.yaml)
records CI ownership, selection mode, and path triggers. Profile code remains
the source of truth for deployment behavior and its exact test list.

## Run selected test cases

```bash
make e2e-test-specific \
  E2E_PROFILE=envoy-ai-gateway \
  E2E_TESTS='chat-completions-request,decision-fallback-behavior'
```

Names must be registered under `e2e/testcases`. A test can be runnable by name
without being part of every profile's default contract.

## Keep an environment for debugging

```bash
make e2e-test-debug E2E_PROFILE=envoy-ai-gateway
```

Or split setup from repeated test runs:

```bash
make e2e-setup E2E_PROFILE=envoy-ai-gateway
make e2e-test-only E2E_PROFILE=envoy-ai-gateway
make e2e-test-only \
  E2E_PROFILE=envoy-ai-gateway \
  E2E_TESTS='chat-completions-request'
make e2e-cleanup
```

Use the same `E2E_CLUSTER_NAME` and profile for each command. `e2e-test-only`
assumes the expected releases, values, images, and Services are already in the
cluster; it does not reconcile drift first.

## Runner options

| Make variable | Default | Effect |
| --- | --- | --- |
| `E2E_PROFILE` | `envoy-ai-gateway` | Registered deployment profile. |
| `E2E_CLUSTER_NAME` | `semantic-router-e2e` | Kind cluster name. |
| `E2E_IMAGE_TAG` | `e2e-test` | Tag used for locally built images. |
| `E2E_KEEP_CLUSTER` | `false` | Preserve the cluster after the run. |
| `E2E_USE_EXISTING_CLUSTER` | `false` | Skip Kind creation and use current access. |
| `E2E_TESTS` | empty | Comma-separated test names; empty uses the profile list. |
| `E2E_PARALLEL` | `false` | Run selected test cases concurrently. |
| `E2E_VERBOSE` | `true` | Print lifecycle and test details. |
| `E2E_SETUP_ONLY` | `false` | Deploy the profile without tests. |
| `E2E_SKIP_SETUP` | `false` | Run against an already deployed profile. |
| `E2E_USE_WORKSPACE_MODELS` | `false` | Mount the workspace `models/` directory into a new Kind cluster. |

Parallel tests must not mutate the same runtime state. Leave parallel mode off
until the selected cases are known to be isolated.

## Profile selection

### Supported Profiles

- **envoy-ai-gateway**: baseline routing, safety, cache, and decision contracts.
- **external-gateway-responses**: ExtProc-only Responses create, get, and conversation chaining with external gateway-owned dispatch.
- **dashboard**: dashboard API, validation, and routing-authoring contracts.
- **aibrix**: AIBrix gateway and control-plane integration.
- **routing-strategies**: keyword, entropy, and fallback routing.
- **dynamic-config**: CRD-driven routing and embedding signals.
- **multimodal-routing**: image-modality embedding routing.
- **remote-embedding**: OpenAI-compatible remote embedding providers.
- **category-remote-backend**: shared remote category `http_classify` backend.
- **llm-d**: llm-d inference-gateway health and router smoke coverage.
- **looper**: deterministic Looper algorithm contracts.
- **istio**: sidecar, mTLS, and tracing behavior.
- **agentgateway**: agentgateway routing and ExtProc policy enforcement.
- **production-stack**: HA, load balancing, failover, and load checks.
- **ml-model-selection**: trained model-selector behavior.
- **multi-endpoint**: environment policy across several backends.
- **authz-rbac**: authorization routing and rate-limit behavior.
- **streaming**: streamed request bodies and cache round trips.
- **anthropic-shim**: affected-change Anthropic backend and cross-protocol matrix coverage.
- **response-api**: affected-change memory-backed Responses API and cross-protocol matrix coverage.
- **route-action**: decision route action for detected prompt attacks and benign traffic.
- **response-api-redis**: manual Redis persistence and TTL coverage.
- **response-api-redis-cluster**: manual Redis Cluster persistence and TTL coverage.
- **router-replay**: manual management-boundary and restart-recovery coverage.
- **dynamo**: manual NVIDIA Dynamo batching and GPU health coverage.
- **vectorstore-registry**: manual metadata restart-recovery coverage.
- **rag-hybrid-search**: manual Llama Stack hybrid-search coverage.
- **hallucination**: manual fact-check gating and warning behavior.
- **jailbreak-onerror**: manual PromptGuardConfig.OnError coverage against an unreachable classifier endpoint.
- **response-jailbreak**: response-direction jailbreak signal and response_jailbreak plugin coverage for LLM output carrying jailbreak content past the classifier's sequence window, and the streamed-response pass-through contract.

### Coverage Ownership Matrix

| Selection | Meaning | Source of truth |
| --- | --- | --- |
| Default local | Runs when no profile is specified | `default_local: true` in the test-domain registry |
| Full CI | Runs in the complete E2E matrix | `full_ci: true` in the test-domain registry |
| Affected | Selected when owned paths change | `selection: pr` and `paths` in the test-domain registry |
| Manual only | Requires explicit selection and profile prerequisites | `selection: manual` in the test-domain registry |

[`tools/agent/test-domain-registry.yaml`](../tools/agent/test-domain-registry.yaml)
owns the exact selection mode, path triggers, and coverage role for every entry.
“Manual” describes lifecycle and prerequisites; it is not evidence that the
profile passed in another environment.

The `response-api` and `anthropic-shim` affected-change profiles jointly own the
three native protocol backends used by the pairwise codec matrix. Their default
contracts exercise Chat Completions, Responses, and Messages clients against each
backend in buffered and streaming modes: 3 client protocols x 3 backend protocols
x 2 response modes, for 18 required end-to-end cells. Each cell validates the
client-native response envelope or SSE sequence, the terminal event, translated
backend output, and the absence of leaked backend wire shapes. The same profiles
also cover tool-call lifecycles, structured JSON Schema output, provider transport
errors, incomplete streams, and midstream failures.

## Add or change a profile

1. Add or update a package under `e2e/profiles/<name>`.
2. Implement `framework.Profile`: name, setup, teardown, test list, and Service
   access.
3. Register it in `e2e/profiles/all/imports.go`.
4. Add its ownership and selection mode to
   `tools/agent/test-domain-registry.yaml`.
5. Reuse test cases where the contract is shared; add a new test only for a new
   externally visible behavior.
6. Add deterministic assertions. A request that merely returned any response
   is not sufficient evidence for routing, safety, cache, or fallback behavior.

For test-case boundaries, read [`testcases/AGENTS.md`](testcases/AGENTS.md).

## Validate framework changes

```bash
make build-e2e
(cd e2e && go test ./...)
make agent-report ENV=cpu CHANGED_FILES='e2e/...'
```

Then run the smallest affected profile. Use
`make agent-e2e-affected CHANGED_FILES='...'` when the repository harness can
resolve the profile set from changed paths.

## Diagnose a failed run

The runner prints the failing test and writes reports even when the run fails.
With a preserved cluster:

```bash
kubectl get pods --all-namespaces
kubectl get events --all-namespaces --sort-by=.lastTimestamp
kubectl get gateway,httproute --all-namespaces
```

Separate setup failure, test assertion failure, and teardown failure. Keep the
profile only long enough to inspect it, then delete the named Kind cluster:

```bash
make e2e-cleanup E2E_CLUSTER_NAME=semantic-router-e2e
```
