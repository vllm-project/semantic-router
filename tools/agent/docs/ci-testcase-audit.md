# CI testcase audit

This audit accompanies the [testing strategy](testing-strategy.md). It compares
the 2026-10-18 baseline `1c629c80b383d7e53e01f266430d0fa41c5d9bf6` with the
CI consolidation change. It is evidence about test value, not a second
execution allowlist.

The canonical baseline, Dashboard and provider groups are reviewed below.
Companion audits cover every remaining registered case, including its helpers,
acceptance wrapper, limitations and disposition:

- [Protocol, Responses and streaming cases](ci-protocol-testcase-audit.md)
- [Routing and deployment cases](ci-routing-testcase-audit.md)
- [Runtime, state and model cases](ci-runtime-testcase-audit.md)

Together these reviews cover all 238 surviving registrations and the retired
cases. A static assertion review is not a live passing result. All 17 previously
unreachable registrations have been retired; the execution graph now requires
every registered case to belong to a profile, without a debt exception table.

## Sources and counts

The owners are [testmatrix](../../../e2e/pkg/testmatrix/testcases.go), each
profile's `GetTestCases`, and [testcase registration](../../../e2e/pkg/testcases/registry.go).
Generate the expanded graph without starting a cluster:

```bash
cd e2e
go run ./cmd/e2e-audit > /tmp/e2e-inventory.json
```

| Inventory | Before | After | Meaning |
| --- | ---: | ---: | --- |
| Baseline full | 44 | 41 | Three redundant weak cases retired |
| Baseline standard | 44 | 39 | Same retained functional cases; two stress cases reserved for full scope |
| Dashboard | 16 | 16 | Product API/state/recovery contracts |
| Provider protocols | 25 | 25 | Native Anthropic backend and client conversion contracts |
| Unique cases across those three groups | 83 | 80 | 82 final group entries; two intentional cross-backend repeats |
| Global testcase registrations | 278 | 238 | Forty legacy or retired-integration registrations removed |

`RouterSmoke` contains only `chat-completions-request`, already counted above.
`chat-completions-request` and `anthropic-messages-streaming` run in both baseline
and provider groups against different backend protocols. A testcase count is
not a request count, assertion count, Go subtest count or number of jobs per PR.

The 238 registrations are the repository's available integration coverage,
not the default PR workload. The fixed full-CPU inventory selects 11 profiles:
**82 profile/case entries representing 79 distinct case names**. A changed
surface can add an optional profile; ordinary PRs select affected contracts
and require committer authorization for expensive execution.

| Default full-CPU integration family | Profile/case entries | Boundary protected |
| --- | ---: | --- |
| Canonical router | 41 | Routing, signals, request mutation, safety, cache and accounting |
| Dashboard | 16 | Configuration, authorization, deployment and recovery |
| Streaming | 9 | Stream transport, safety, replay and terminal correctness |
| Standalone | 7 | Direct serving, deployment, configuration and failover |
| Routing errors | 2 | Router-origin failure contracts |
| Multimodal, local classifier and remote embedding | 4 | Model-adapter and routing wiring |
| Vela Omni, Halu and Shield | 3 | Published model identity and positive/negative probes |

The 17 unreachable cases removed in the final cleanup were nine old selector
cases, five MCP cases, one superseded RBAC case, one OpenAI RAG case and one
rate-limit case. Their exclusive helpers, data and duplicate RAG manual script
are removed as well. Existing selector/learning/identity assertions and small
MCP transport, search and post-auth quota tests preserve the useful current
boundaries; the companion audits identify each replacement and its limits.
This cleanup removes maintenance debt, not executed runner minutes. The job
and artifact savings come from selection and shared expensive setup, as
quantified in the [cost audit](ci-job-cost-audit.md).

Four selected legacy routing aggregates (embedding, entropy, keyword and
fallback) are also retired, removing 89 probe requests when both affected
profiles run. They accepted one matching observation or inferred missing
evidence; their reports were not useful acceptance gates. Direct classifier,
decision and selector tests remain, together with discriminating routed
contracts. This does not claim equivalent live qualification of every former
natural-language prompt or the old fallback-chain description. The permissive
Responses backend-error case is removed; existing injected-error matrices
require exact status and native error fields.

Model-selection and the shared signal-routing cases reject empty or invalid
corpora; model-selection previously fell back to duplicated built-in data.
Failover requires successful fault injection. Streaming keyword routing requires
all positive controls and a fallback control; large-body transport checks the
provider-observed messages, and Redis profiles require the declared storage
topology. Local negative controls protect these acceptance rules.

Maintained ingress coverage is deliberately limited to Envoy AI Gateway ExtProc
and standalone. Agentgateway, AIBrix, Istio, llm-d and Dynamo have no executable
E2E profiles or manual CI entries; their guides are experimental examples.
Standalone retains seven Kubernetes cases: chart defaults, provider-observed
chat, models, backend fallback, typed Router errors, decision-model propagation,
and configuration rollout. The CLI also exercises explicit standalone startup
without an Envoy container. Neither standalone suite currently qualifies SSE;
the retained streaming profile qualifies the Envoy ExtProc transport.

## Expanded profile inventory

The runtime graph after consolidation contains **42 profiles and 238 registered
cases**. Every selected name is registered, and every registered name is reachable.
The [execution-graph gate](../../../e2e/pkg/verification/gates_test.go) rejects
unreachable registrations instead of allowing a growing debt table. Profile selections overlap,
so this table must not be summed as unique coverage. It is a dated audit snapshot;
`GetTestCases` and the runtime inventory command remain authoritative.

| Profile | Selected cases |
| --- | ---: |
| `authz-rbac` | 3 |
| `category-remote-backend` | 1 |
| `complexity-remote-backend` | 1 |
| `complexity-routing` | 1 |
| `conversation-routing` | 1 |
| `dashboard` | 16 |
| `dynamic-config` | 3 |
| `envoy-ai-gateway` | 41 |
| `external-gateway-responses` | 3 |
| `hallucination` | 2 |
| `jailbreak-onerror` | 1 |
| `local-classifier-backend` | 1 |
| `long-context-routing` | 1 |
| `looper` | 13 |
| `masking` | 2 |
| `ml-model-selection` | 3 |
| `model-runtime` | 12 |
| `model-runtime-real` | 1 |
| `multi-endpoint` | 4 |
| `multimodal-routing` | 2 |
| `pii-remote-backend` | 2 |
| `preference-routing` | 1 |
| `production-stack` | 10 |
| `progress-gate` | 1 |
| `provider-protocols` | 25 |
| `rag-hybrid-search` | 2 |
| `remote-embedding` | 1 |
| `response-api` | 57 |
| `response-api-redis` | 12 |
| `response-api-redis-cluster` | 11 |
| `response-jailbreak` | 3 |
| `route-action` | 2 |
| `router-replay` | 13 |
| `routing-errors` | 2 |
| `routing-strategies` | 4 |
| `standalone` | 7 |
| `streaming` | 9 |
| `structure-routing` | 1 |
| `vectorstore-registry` | 1 |
| `vela-halu` | 1 |
| `vela-omni` | 1 |
| `vela-shield` | 1 |

## Follow the effective assertion

[Register](../../../e2e/pkg/testcases/registry.go) wraps testcase functions with
[profile-scoped acceptance contracts](../../../e2e/pkg/testcases/acceptance_contracts.go).
A function returning `nil` after recording results is not necessarily a false
pass. Baseline floors are domain 60%, cache hits 40%, PII 65%, priority 75%, and
fallback 85%. Jailbreak additionally requires 80% in each attack/non-attack
class and fails request/contract errors directly; its wrapper also checks 70%
aggregate accuracy. Sequential stress requires 95%; progressive 10/20/50
stages require 95/85/60%, with 75% overall. Missing details and invalid counts
fail acceptance. These model/pressure thresholds remain unchanged.

Other profiles do not inherit the baseline wrapper. For example,
`TestWrapWithAcceptanceContractSkipsUnscopedProfiles` explicitly preserves that
behavior for dynamic-config. Production-stack domain/PII and dynamic-config
priority reuse therefore have weaker aggregate requirements; they need their
own calibrated acceptance policy before being advertised as equivalent gates.

## Retire or repair misleading coverage

| Case | Observed weakness | Retained boundary or replacement |
| --- | --- | --- |
| `plugin-config-variations` | Eight fixtures name `response_cache`, while the switch handled `semantic-cache`; they only compared decision names. Ten system-prompt fixtures assigned `prompt_applied` without observing a provider request. An 80% wrapper cannot repair this oracle. | [plugin-request-mutations](../../../e2e/testcases/plugin_request_mutations.go) checks actual messages, request parameters and headers. [semantic-cache](../../../e2e/testcases/cache.go), polarity and exact-negation cases check real cache behavior. |
| `plugin-chain-execution` | `cacheCorrect` was always true; prompt application was inferred from a nonempty decision. The effective check was PII blocking, not chain order. | PII detection, [short-circuit no-dispatch](../../../e2e/testcases/plugin_short_circuit_dispatch.go) with a positive provider control, and [tool-selection](../../../e2e/testcases/tool_selection_e2e.go) provider/PII precedence. |
| `rule-condition-logic` | All six rows were positive OR probes; rule operators/conditions were printed but not independently tested. | `TestDecisionEngine_EvaluateDecisions` in [engine_test.go](../../../src/semantic-router/pkg/decision/engine_test.go) covers full/partial AND, OR and priority. `TestEvaluateDecisionsWithTrace_ANDWithOneFailing` in [trace_test.go](../../../src/semantic-router/pkg/decision/trace_test.go) verifies the negative branch and trace. Routed signal/priority/fallback tests remain. |
| `istio-mtls-verification` | Certificate failures did not clear its success flag; policy inspection did not prove a TLS handshake or peer identity. | Retired with the entire Istio-specific E2E profile. The maintained Envoy ExtProc contract does not qualify Istio transport security. |
| `istio-tracing-observability` | No collector/span acceptance; generic metric text and unchecked request outcomes were insufficient. | Retired with the entire Istio-specific E2E profile. Neither metric presence nor a generic ExtProc request establishes trace-collection coverage. |
| `authz-header-spoofing` | An absent model counted as correct, the fallback read a nonstandard nested model field, and only one case needed to pass. A greeting also selected 7B for a spoofed premium identity. | Require every probe, a nonempty matching top-level model/selected-model header and consistent evidence. A code-request prompt selects 14B if premium headers leak through, making the expected default 7B discriminating. Parser, failure aggregation and probe tests cover these checks. |

The three legacy plugin/rule implementations, their private JSON fixtures,
acceptance entries and profile selections are removed together. Dynamo's
report-only batching, performance and GPU-utilization cases are also retired:
request failures or absent workers could still pass, and they did not measure
the advertised behavior. The remaining Dynamo health/inference smoke cases are also retired with that
integration profile. The maintained request paths are standalone and Envoy
ExtProc; optional deployment examples carry no platform-specific qualification.

The streamed-body safety probes now require the specific PII/jailbreak decision
and refusal text, plus a benign request that reaches the general decision.
The SSE-cache probe now requires an actual cache hit, SSE content type, complete
Chat stream terminal and exact replay of the primed output. A failed streaming
request can no longer pass on an unrelated non-streaming cache hit. Negative
unit cases exercise wrong blockers, universal blocking, invalid bodies,
truncated/duplicate terminals and altered replay text.

## Repaired integration cases

| Case | Effective assertion and disposition |
| --- | --- |
| [authz-header-spoofing](../../../e2e/testcases/authz_header_spoofing.go) | All four probes require the observed default model, reject missing/conflicting evidence and use a prompt that would select the premium model if identity spoofing succeeded. Keep the trust-boundary regression. |

## Canonical cases

Each row links to the implementation and its helpers. Retaining a case does not
require a separate job, image download or cluster; preserve its result identity
when sharing execution setup.

### Baseline router

| Case | Effective assertion and disposition |
| --- | --- |
| [chat-completions-request](../../../e2e/testcases/chat_completions_request.go) | HTTP 200 deployment smoke. Keep once per distinct deployment, sharing setup with stronger cases. |
| [anthropic-messages-request](../../../e2e/testcases/anthropic_messages_request.go) | HTTP 200, nonempty JSON and selected decision/model. Some overlap with response-shape; share execution setup. |
| [anthropic-messages-protocol-headers](../../../e2e/testcases/anthropic_messages_headers.go) | Upstream recipe, Anthropic client/OpenAI backend headers and absence of loss warnings. Keep the conversion boundary. |
| [anthropic-messages-response-shape](../../../e2e/testcases/anthropic_messages_response_shape.go) | Message ID, type, role, content, stop reason and usage shape. Keep the client response contract. |
| [anthropic-messages-streaming](../../../e2e/testcases/anthropic_messages_streaming.go) | Required SSE events, order and exactly one message_stop. Reuse against different backends exercises different codecs. |
| [apiserver-runtime-config-endpoints](../../../e2e/testcases/apiserver_runtime_config_endpoints.go) | Public model entrypoint, loaded decisions and model inventory state. Keep API wiring; this is not model accuracy. |
| [apiserver-classification-endpoints](../../../e2e/testcases/apiserver_classification_endpoints.go) | Config decision count agrees with metrics and combined classification fields exist. Keep as an API shape smoke. |
| [llm-classifier-distribution-routing](../../../e2e/testcases/llm_classifier_distribution_routing.go) | Four positive/negative decision-match probes. Keep semantic routing wiring. |
| [sequence-classifier-routing](../../../e2e/testcases/sequence_classifier_routing.go) | Positive and negative classifier decision probes. Keep. |
| [chat-completions-stress-request](../../../e2e/testcases/chat_completions_stress_request.go) | 200 sequential requests; baseline wrapper requires 95% HTTP success, with 30-second request timeout. Full scope only; latency is diagnostic. |
| [domain-classify](../../../e2e/testcases/domain_classify.go) | Real predicted categories versus fixture, with a 60% baseline accuracy floor. Other profiles do not inherit this floor. |
| [semantic-cache](../../../e2e/testcases/cache.go) | Per-request similarity, failed priming detection and cancellation without cache publication; baseline also requires 40% hits. Keep. |
| [semantic-cache-polarity](../../../e2e/testcases/cache_polarity.go) | Opposite meaning misses while a paraphrase hits the selected cache policy; empty fixtures fail. Keep the negative control. |
| [exact-cache-multilingual-negation](../../../e2e/testcases/cache_exact_negation.go) | German/Chinese identical requests hit and negations miss the exact-cache recipe. Keep. |
| [pii-detection](../../../e2e/testcases/pii_detection.go) | Observed blocking versus expected outcomes; baseline requires 65% detection accuracy. Preserve the calibrated floor. |
| [pii-entity-offsets](../../../e2e/testcases/pii_entity_offsets.go) | Unicode code-point slices equal entity text, including an entity at position zero. Keep the tokenizer/API boundary. |
| [pii-long-text](../../../e2e/testcases/pii_long_text.go) | An entity beyond the initial sequence window is detected with valid offsets. A mocked tokenizer cannot replace this. |
| [security-long-text](../../../e2e/testcases/security_long_text.go) | Benign long filler stays benign; an appended attack is detected. Keep both controls. |
| [security-window-provenance](../../../e2e/testcases/security_window_provenance.go) | Inventory-driven window policy, multiple-window provenance or absence of false window metadata for whole-prompt models. Keep. |
| [jailbreak-detection](../../../e2e/testcases/jailbreak_detection.go) | Each attack/non-attack class needs 80% accuracy; request/contract errors fail directly. Baseline also checks a 70% aggregate floor. |
| [decision-priority-selection](../../../e2e/testcases/decision_priority.go) | Four selected-decision probes, with a 75% baseline floor. Dynamic-config reuse has no equivalent wrapper; record that limitation. |
| [plugin-request-mutations](../../../e2e/testcases/plugin_request_mutations.go) | Actual provider messages, token/n caps, removed parameters and header mutations. Keep; replaces inferred plugin effects. |
| [tool-selection](../../../e2e/testcases/tool_selection_e2e.go) | Every subcase must pass provider tool passthrough/filter/removal and PII no-dispatch checks. Optional debug fields are supplemental. |
| [decision-fallback-behavior](../../../e2e/testcases/decision_fallback.go) | Explicit match/no-match fixture plus provider model/message observations; baseline floor is 85%. Keep. |
| [chat-completions-progressive-stress](../../../e2e/testcases/chat_completions_progressive_stress.go) | 10/20/50 concurrent requests require 95/85/60% per stage and 75% overall. Full scope only; no latency regression threshold. |
| [protocol-codec-openai-regression](../../../e2e/testcases/protocol_codec_openai_regression.go) | Native Chat HTTP/JSON and selected route smoke. Some overlap with chat smoke; share the worker, retain its receipt. |
| [retention-directive](../../../e2e/testcases/retention_directive.go) | A pinned decision emits exact retention headers and omits unset fields. Keep. |
| [looper-latency-token-headers](../../../e2e/testcases/looper_latency_token_headers.go) | A pinned Looper decision emits positive latency/token fields and total equals prompt plus completion. Keep accounting, not timing accuracy. |
| [entrypoint-recipe-routing](../../../e2e/testcases/entrypoint_recipes.go) | Explicit entrypoint selects its recipe/model; default remains isolated; model listing includes entrypoints. Keep. |
| [unknown-model-error-codes](../../../e2e/testcases/routing_error_codes.go) | Exact Router 400 error envelope/code despite a global default. Keep the failure boundary. |
| [chat-completions-structured-output](../../../e2e/testcases/chat_completions_structured_output.go) | A json_schema request routes and returns Chat choices. This is routing smoke; exact schema forwarding is separately unit-tested. |
| [plugin-short-circuit-no-dispatch](../../../e2e/testcases/plugin_short_circuit_dispatch.go) | Provider observation is absent for blocked requests and present for a positive control. Keep the actual dispatch boundary. |
| [streaming-usage-settlement](../../../e2e/testcases/streaming_usage_settlement.go) | Provider receives forced include_usage; clients receive usage only when requested; SSE terminals and metrics settlement are checked. Keep. |
| [session-telemetry-metrics](../../../e2e/testcases/session_telemetry_e2e.go) | Only HTTP success and metric-name presence. Treat as exporter smoke; it does not prove a request-specific counter increment. |
| [session-pricing-chat-completions](../../../e2e/testcases/session_pricing_e2e.go) | Token/cost metric names exist, without exact amount or request attribution. Retain as smoke; consider consolidating with telemetry. |
| [session-pricing-response-api](../../../e2e/testcases/session_pricing_e2e.go) | The Responses entrypoint succeeds and exposes token/cost metric names. It does not establish price calculation correctness. |
| [event-routing](../../../e2e/testcases/event_routing.go) | Shared helper checks every positive/negative decision and matched event rule. Keep. |
| [language-routing](../../../e2e/testcases/language_routing.go) | Every decision and language-rule expectation must pass. Keep. |
| [reask-routing](../../../e2e/testcases/reask_routing.go) | Every decision and reask-rule expectation must pass. Keep. |
| [action-routing](../../../e2e/testcases/action_routing.go) | Every decision and action-rule expectation must pass. Keep. |
| [context-safety-overlap](../../../e2e/testcases/context_safety_overlap.go) | Safety wins when a long-context signal overlaps jailbreak/PII conditions. Keep the precedence boundary. |

### Dashboard

| Case | Effective assertion and disposition |
| --- | --- |
| [dashboard-health](../../../e2e/testcases/dashboard_health.go) | Health HTTP/response shape smoke. Share the existing Dashboard environment. |
| [dashboard-status](../../../e2e/testcases/dashboard_status.go) | Status and component shape smoke. It is not a substitute for component behavior. |
| [dashboard-openapi](../../../e2e/testcases/dashboard_openapi.go) | Published OpenAPI route contract. Keep API registration consistency. |
| [dashboard-capabilities](../../../e2e/testcases/dashboard_capabilities.go) | Caller capability states and dependency-probe behavior. Keep authorization and availability distinctions. |
| [dashboard-route-bound-authorization](../../../e2e/testcases/dashboard_route_bound_authorization.go) | Invitations, role-bound APIs and immediate session revocation. Keep. |
| [dashboard-config-read](../../../e2e/testcases/dashboard_config_read.go) | Canonical JSON/YAML and expected model/decision identity, beyond valid syntax. Keep. |
| [dashboard-deploy-preview](../../../e2e/testcases/dashboard_deploy_preview.go) | Preview response and configuration-processing boundary. Keep separately from applied deployment. |
| [dashboard-config-versions](../../../e2e/testcases/dashboard_config_versions.go) | Version API/record contract. Share setup; retain independent evidence. |
| [dashboard-deploy-invalid-yaml](../../../e2e/testcases/dashboard_deploy_invalid_yaml.go) | Invalid YAML rejection. This differs from semantically invalid configuration. |
| [dashboard-deploy-safe-failure](../../../e2e/testcases/dashboard_deploy_safe_failure.go) | Invalid semantic deployment leaves active configuration/versions intact and serving. Keep. |
| [dashboard-deploy-rollback](../../../e2e/testcases/dashboard_deploy_rollback.go) | Exact pre-deploy state is restored and deploy/rollback operations are recorded. Keep. |
| [dashboard-status-active-config](../../../e2e/testcases/dashboard_status_active_config.go) | Status version/hash agrees with the configuration served after deploy and rollback. Keep. |
| [dashboard-logs-diagnostics](../../../e2e/testcases/dashboard_logs_diagnostics.go) | Unsupported-log contract, invalid component rejection, rejected-deploy audit increment and unchanged active state. Keep. |
| [dashboard-anonymous-access-denied](../../../e2e/testcases/dashboard_anonymous_access.go) | Anonymous/forged sessions get 401; configuration, versions and active projection remain unchanged. Keep. |
| [dashboard-sr-bench](../../../e2e/testcases/dashboard_sr_bench.go) | Frozen two-case plan, scoring/accounting, idempotency and cancellation without dispatching queued cases. Keep. |
| [dashboard-restart-recovery](../../../e2e/testcases/dashboard_restart_recovery.go) | Dashboard and independent worker restarts preserve sessions/workflows and benchmark evidence. Keep fault recovery. |

### Provider protocols

| Case | Effective assertion and disposition |
| --- | --- |
| [prompt-compression-json-routing](../../../e2e/testcases/prompt_compression_json_routing.go) | Short marker control versus long JSON/code routing, while the provider receives the original prompt. Keep. |
| [chat-completions-request](../../../e2e/testcases/chat_completions_request.go) | HTTP 200 deployment smoke. Keep once per distinct deployment, sharing setup with stronger cases. |
| [anthropic-messages-cache-cycle](../../../e2e/testcases/anthropic_messages_cache_cycle.go) | Cache-write and subsequent cache-read usage fields. Keep provider cache semantics. |
| [anthropic-chat-cache-control](../../../e2e/testcases/anthropic_chat_cache_control.go) | Buffered/streamed provider marker ownership and cached-token projection into Chat. Keep. |
| [anthropic-prompt-cache-policy](../../../e2e/testcases/anthropic_prompt_cache_policy.go) | Insertion, empty instruction, caller precedence and disabled policy observed at the provider. Keep. |
| [anthropic-messages-stop-sequence](../../../e2e/testcases/anthropic_messages_stop_sequence.go) | Provider stop_sequence maps to Messages stop_reason. Keep. |
| [anthropic-chat-stop-sequence](../../../e2e/testcases/anthropic_chat_stop_sequence.go) | The same terminal maps to Chat finish_reason=stop in buffered and streamed responses. Keep the client distinction. |
| [anthropic-messages-streaming](../../../e2e/testcases/anthropic_messages_streaming.go) | Required SSE events, order and exactly one message_stop. Reuse against different backends exercises different codecs. |
| [anthropic-chat-completions-streaming](../../../e2e/testcases/anthropic_response_api_streaming.go) | Chat SSE shape/terminal without leaking Messages frames. Keep. |
| [anthropic-response-api-buffered](../../../e2e/testcases/anthropic_response_api_streaming.go) | Buffered Responses object/output shape against an Anthropic backend. Keep. |
| [anthropic-response-api-streaming](../../../e2e/testcases/anthropic_response_api_streaming.go) | Responses SSE output and terminal against an Anthropic backend. Keep independently of buffered handling. |
| [protocol-codec-anthropic-backend-buffered-matrix](../../../e2e/testcases/protocol_codec_responses_backend.go) | Three client protocols, client content/shape and actual provider request format. Share one profile execution. |
| [protocol-codec-anthropic-backend-streaming-matrix](../../../e2e/testcases/protocol_codec_responses_backend.go) | Three streaming client protocols, output/terminals and provider requests. Buffered coverage cannot replace it. |
| [protocol-codec-anthropic-response-diagnostics](../../../e2e/testcases/protocol_codec_response_decorations.go) | Native diagnostics survive; unsupported cross-protocol fields are removed with loss warnings. Keep. |
| [protocol-codec-responses-verbosity-anthropic](../../../e2e/testcases/protocol_codec_responses_custom_tools.go) | Unsupported verbosity is absent at the provider and reported as dropped, for both response modes. Keep. |
| [protocol-codec-prompt-cache-key-anthropic](../../../e2e/testcases/protocol_codec_prompt_cache_key.go) | Chat/Responses and buffered/streamed requests drop unsupported cache keys with diagnostics. Keep. |
| [protocol-codec-anthropic-per-message-effort-backend](../../../e2e/testcases/protocol_codec_anthropic_per_message_effort.go) | Provider per-turn effort and beta header, plus unsupported-projection removal/warnings. Keep. |
| [protocol-codec-zero-penalty-anthropic](../../../e2e/testcases/protocol_codec_zero_penalty_anthropic.go) | Zero-valued unsupported penalty controls do not leak to the provider. Keep the compatibility regression. |
| [protocol-codec-anthropic-backend-tool-lifecycle](../../../e2e/testcases/protocol_codec_responses_backend.go) | Three clients preserve tool calls and second-turn results in buffered/streamed modes. Keep identity and argument checks. |
| [protocol-codec-anthropic-backend-structured-output](../../../e2e/testcases/protocol_codec_structured_output.go) | Provider-bound schema and client structured-output contract. Stronger than the baseline routing-only schema smoke. |
| [protocol-codec-anthropic-backend-error-matrix](../../../e2e/testcases/protocol_codec_responses_backend.go) | Provider failure remains failure in every client protocol. Keep. |
| [protocol-codec-anthropic-backend-incomplete-stream-matrix](../../../e2e/testcases/protocol_codec_responses_backend.go) | Early stream closure produces a failure terminal without success. Keep. |
| [protocol-codec-anthropic-backend-midstream-error-matrix](../../../e2e/testcases/protocol_codec_responses_backend.go) | Partial output precedes exactly one error, with no later success terminal. Keep. |
| [protocol-codec-anthropic-backend-agent-client-replay](../../../e2e/testcases/protocol_codec_agent_client_replay.go) | Stateful agent/tool interaction replay across client protocols. A single successful request cannot replace it. |
| [public-listener-empty-body-rejected](../../../e2e/testcases/public_listener_empty_body.go) | Seven public paths require Router-owned 400 envelopes and no leaked upstream headers. Keep. |

## Other profile families

The following summarize the companion per-case reviews linked above. Those
reviews retain the individual assertions and limitations behind each family.

| Family | Actual boundary and disposition |
| --- | --- |
| Model runtime: 12 fixture cases, 1 published-model case | [Tiny random-weight packages](../../../e2e/profiles/model-runtime/runtime_with_fixtures.py) exercise readiness ordering, independent workers, RPC outputs, bundles, supervision and load-retry isolation without downloads. [Real routing](../../../e2e/testcases/model_runtime_real.go) checks five clear requests with published Kai/Vela and compares Router signals to Kai's own answer. Keep both; fixtures qualify process integration, not model accuracy. |
| Published Vela: Shield, Halu, Omni | [Shield](../../../e2e/testcases/vela_shield.go) requires unsafe/benign/over-refusal verdicts. [Halu](../../../e2e/testcases/vela_halu.go) checks supported/contradictory answers, Unicode spans and input-budget rejection. [Omni](../../../e2e/testcases/vela_omni.go) checks dimensions, normalized vectors, reference cosines, invalid input and recipe isolation. Keep real-model qualification separate from fast units. |
| Responses and protocol extensions | [response-api](../../../e2e/profiles/response-api/profile.go) selects CRUD, chaining, failure, image and multi-client protocol cases. The reviewed [backend matrix helpers](../../../e2e/testcases/protocol_codec_responses_backend.go) check provider requests, client output and incomplete/midstream failures; reuse across Chat, Responses and Anthropic is intentional. Redis variants add persistence/restart boundaries. |
| Streaming and privacy | [Body limits](../../../e2e/testcases/streaming_body_limits.go) check 413/408 and exact error codes before routing. [Masking](../../../e2e/testcases/masking_cross_protocol.go) observes provider input without raw PII, requires placeholders and rejects unavailable classifiers with 503. Streamed safety now checks the exact decision/refusal and a benign control; no-dispatch is still owned by the separate provider-observation cases. |
| Routing and selection | Signal families compare positive/negative matched rules and decisions. [Decision-scoped multi-factor](../../../e2e/testcases/decision_scoped_multi_factor.go) requires distinct quality/cost model choices. [Random](../../../e2e/testcases/decision_scoped_random.go) checks eligible candidates and algorithm wiring; statistical uniformity belongs in selector units. Shared selection infrastructure does not make these distinct policy boundaries redundant. |
| Replay, shadow and state | [Replay restart recovery](../../../e2e/testcases/router_replay_restart_recovery.go) confirms a record in Postgres and provider-observed dispatch hash/length, then preserves it after restart. [Shadow capture](../../../e2e/testcases/shadow_dispatch_capture.go) checks candidate metadata, disabled response-body capture and failure reasons. Keep stateful receipts; do not substitute generic HTTP smoke. |
| Deployment entrypoints | Maintain Envoy AI Gateway for ExtProc and the standalone Router listener. Retire agentgateway, AIBrix, Istio, llm-d and Dynamo profiles, their exclusive cases and fixtures. Their deployment guides are experimental examples; no generic ExtProc result qualifies their control planes. |
| Production stack: 10 cases | Failover deletes a real backend pod and requires a backend success floor. Load balancing requires endpoints and backend-routed requests, not uniform distribution. [Resource monitoring](../../../e2e/testcases/resource_utilization_monitoring.go) proves a nonempty Prometheus series; [throughput](../../../e2e/testcases/performance_throughput.go) rejects request errors without a comparative performance threshold. Use full/manual qualification and describe the limits accurately. |
| Operator | Keep one memory-backed reconciliation and provider-observed routed request. Storage sample conversion stays in direct source tests; backend data behavior belongs in storage contracts. Removed storage deployments did not assert hits, persistence or hybrid-tier behavior. |
| Local product and benchmark contracts | The development launcher smoke now shares the existing CLI worker, but its separate receipt must be present and passed. Benchmark units check fixture independence, timers, cleanup and model identity without checkpoint inference. Identity now includes engine arguments, preventing unlike executions from sharing a baseline. Live inference and pressure belong in explicit qualification. |

### Remaining work

- Session telemetry/pricing cases check metric names rather than this request's
  increments, labels or exact cost. Consolidate their requests and add accounting
  assertions before treating them as pricing correctness tests.
- Give non-baseline reuse an explicit acceptance policy; do not silently apply a
  baseline model-quality threshold to a different deployment or model.
- Keep remaining limitations explicit: response framing does not prove delivery
  latency, namespace CPU series do not measure Router efficiency, and model
  membership does not prove algorithm quality. Manual execution does not change
  those evidence boundaries.
- Keep the distinction between deleting a redundant oracle, sharing a worker,
  and moving expensive qualification to full/manual execution. Per-case missing,
  failed and skipped results must remain visible in the Gate.

### Validation boundary

Local validation builds the E2E binary and runs framework/profile/testcase units
and runtime inventory drift gates, plus benchmark identity and smoke-receipt
unit tests. These checks do not establish that a live Kind deployment, published
model or pressure run has passed. Those results come from the selected CI
qualification receipts.
