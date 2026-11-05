package headers

import (
	"slices"
	"strings"
)

// Direction is the set of hops a header travels on, relative to the Router.
type Direction uint8

const (
	// FromClient headers arrive on a client's request, from the client or
	// from the gateway and authenticator in front of the Router.
	FromClient Direction = 1 << iota
	// ToClient headers are on the Router's response to a client.
	ToClient
	// ToProvider headers are on a request the Router sends to a provider.
	ToProvider
	// FromProvider headers arrive on a provider's response to the Router.
	FromProvider
)

var directionLabels = []struct {
	direction Direction
	label     string
}{
	{FromClient, "request"},
	{ToClient, "response"},
	{ToProvider, "provider request"},
	{FromProvider, "provider response"},
}

// String names the hops, such as "request, provider request".
func (d Direction) String() string {
	labels := make([]string, 0, len(directionLabels))
	for _, entry := range directionLabels {
		if d&entry.direction != 0 {
			labels = append(labels, entry.label)
		}
	}
	return strings.Join(labels, ", ")
}

// Surface says who may see a header.
type Surface string

const (
	// SurfacePublic headers ride on the default response.
	SurfacePublic Surface = "public"
	// SurfaceDebug headers ride on a response only when the caller is
	// allowed the debug headers.
	SurfaceDebug Surface = "debug"
	// SurfaceControl headers are sent by clients to steer or identify a
	// request.
	SurfaceControl Surface = "control"
	// SurfaceInternal headers pass between the Router, the gateway and
	// authenticator in front of it, and providers. They are not part of the
	// client contract, and no response the Router writes carries them.
	SurfaceInternal Surface = "internal"
)

// Sensitivity says what a header's value reveals.
type Sensitivity string

const (
	SensitivityNone Sensitivity = "none"
	// SensitivityIdentity values name an authenticated caller.
	SensitivityIdentity Sensitivity = "identity"
	// SensitivitySecret values are credentials.
	SensitivitySecret Sensitivity = "secret"
)

// Header records one header of the Router's contract.
type Header struct {
	// Name is the lowercase field name.
	Name        string
	Direction   Direction
	Surface     Surface
	Sensitivity Sensitivity
	// Owner is the feature that reads or writes the header.
	Owner string
	// Version is the release the header first shipped in, such as "v0.3".
	Version string
	// Doc says what the header carries, in one sentence.
	Doc string
}

// Registry lists every header the Router reads or writes: each x- header and
// each provider credential header. Standard HTTP fields the Router handles
// only as HTTP, such as content-type or retry-after, and pseudo-headers are
// not part of its contract.
func Registry() []Header { return slices.Clone(registry) }

// Lookup returns the registered header named name, in any letter case.
func Lookup(name string) (Header, bool) {
	header, ok := registryByName[strings.ToLower(strings.TrimSpace(name))]
	return header, ok
}

// KeptFromProviders are the headers a client request may carry that no
// provider receives: every registered header that arrives from clients and is
// not registered to reach providers. The ingress controls are left out: the
// native gateway's edge drops them before routing, and Envoy owns them behind
// an Envoy gateway. Both gateway modes remove these headers from every
// provider-bound request.
func KeptFromProviders() []string {
	var names []string
	for _, header := range registry {
		if header.Direction&FromClient != 0 && header.Direction&ToProvider == 0 && header.Owner != "ingress" {
			names = append(names, header.Name)
		}
	}
	return names
}

var registryByName = func() map[string]Header {
	byName := make(map[string]Header, len(registry))
	for _, header := range registry {
		byName[header.Name] = header
	}
	return byName
}()

var registry = []Header{
	// The response contract.
	{VSRSchemaVersion, ToClient, SurfacePublic, SensitivityNone, "response_contract", "v0.4", "Response-header contract revision; currently `2`."},
	{VSRResponsePath, ToClient, SurfacePublic, SensitivityNone, "response_contract", "v0.4", "How the response was produced: `upstream`, `cache`, `fast_response`, `looper`, `image_generation`, `blocked`, `rate_limited`, `error` or `fallback`."},
	{VSRConfigVersion, ToClient, SurfacePublic, SensitivityNone, "config", "v0.5", "Version of the configuration snapshot that served the request."},
	{VSRConfigHash, ToClient, SurfacePublic, SensitivityNone, "config", "v0.4", "SHA-256 of the active runtime document, on routing previews, configuration activations and config-bound evaluation requests."},

	// Final routing facts.
	{VSRSelectedRecipe, ToClient, SurfacePublic, SensitivityNone, "routing", "v0.4", "Recipe the effective entrypoint selected; omitted for concrete backend models."},
	{VSRSelectedDecision, ToClient, SurfacePublic, SensitivityNone, "routing", "v0.1", "Decision the decision engine selected; omitted when no decision matched."},
	{VSRSelectedConfidence, ToClient, SurfacePublic, SensitivityNone, "routing", "v0.2", "Model-derived score of the selected decision, such as `0.9100`."},
	{VSRSelectedAlgorithm, ToClient, SurfacePublic, SensitivityNone, "routing", "v0.4", "Model-selection algorithm used after the decision matched, such as `static`."},
	{VSRSelectedModel, ToClient, SurfacePublic, SensitivityNone, "routing", "v0.1", "Model alias the Router selected."},
	{VSRRoutingLatencyMs, ToClient, SurfacePublic, SensitivityNone, "routing", "v0.4", "Time spent choosing the model, in milliseconds, such as `0.412`."},
	{VSRAppliedUnknownPolicy, ToClient, SurfacePublic, SensitivityNone, "routing", "v0.4", "Decisions whose unknown result `rules.on_unknown` resolved, as `decision=policy` pairs; also on the `fail_request` 503."},
	{VSRFallbackAttempts, ToClient, SurfacePublic, SensitivityNone, "fallback", "v0.4", "Attempts cross-model fallback made, the primary included, when a fallback candidate served the response."},
	{RouterReplayID, ToClient, SurfacePublic, SensitivityNone, "router_replay", "v0.2", "Router Replay record id, when the record was persisted."},
	{VSREffectiveInputTokens, ToClient, SurfacePublic, SensitivityNone, "automatic_output", "v0.4", "Rendered input tokens of the final automatic-output dispatch, chat template included."},
	{VSREffectiveMaxOutputTokens, ToClient, SurfacePublic, SensitivityNone, "automatic_output", "v0.4", "Output token limit of the final automatic-output dispatch, reasoning included."},
	{VSRCost, ToClient, SurfacePublic, SensitivityNone, "cost", "v0.4", "Usage priced with the served model's configured pricing; buffered responses only."},
	{VSRCostCurrency, ToClient, SurfacePublic, SensitivityNone, "cost", "v0.4", "Currency of `x-vsr-cost`, from `pricing.currency`."},
	{VSRCacheHit, ToClient, SurfacePublic, SensitivityNone, "response_cache", "v0.1", "`true` when the response cache served the response."},
	{VSRFastResponse, ToClient, SurfacePublic, SensitivityNone, "fast_response", "v0.2", "`true` when the `fast_response` plugin answered without a provider call."},
	{VSRClientProtocol, ToClient, SurfacePublic, SensitivityNone, "protocol", "v0.4", "Client wire protocol, such as `anthropic`; on cross-protocol responses and with the debug headers."},
	{VSRUpstreamProtocol, ToClient, SurfacePublic, SensitivityNone, "protocol", "v0.4", "Provider wire protocol; on cross-protocol responses and with the debug headers."},
	{VSRProtocolWarnings, ToClient, SurfacePublic, SensitivityNone, "protocol", "v0.4", "Protocol translation warnings, as comma-separated `severity;reason;field` entries."},
	{VSRResponseWarnings, ToClient, SurfacePublic, SensitivityNone, "response_warnings", "v0.4", "Response warning codes, in order: `hallucination`, `unverified_factual`, `response_jailbreak`."},
	{VSRModelUsage, ToClient, SurfacePublic, SensitivityNone, "benchmark", "v0.4", "Per-model usage receipt of a config-bound evaluation request."},
	{VSRInferenceCallCount, ToClient, SurfacePublic, SensitivityNone, "benchmark", "v0.4", "Provider calls a config-bound evaluation request made."},
	{RateLimitLimit, ToClient, SurfacePublic, SensitivityNone, "rate_limit", "v0.2", "Limit of the rate-limit window that rejected the request."},
	{RateLimitRemaining, ToClient, SurfacePublic, SensitivityNone, "rate_limit", "v0.2", "Requests left in that window."},
	{RateLimitReset, ToClient, SurfacePublic, SensitivityNone, "rate_limit", "v0.2", "Unix time the window resets at."},
	{RequestID, FromClient | ToClient | ToProvider, SurfacePublic, SensitivityNone, "observability", "v0.1", "Request correlation id, forwarded to providers; the native gateway adds one when absent, and the management API echoes it."},
	{XContentTypeOptions, ToClient, SurfacePublic, SensitivityNone, "router_replay", "v0.4", "`nosniff` on Router Replay management responses."},
	{EnvoyUpstreamServiceTime, ToClient, SurfacePublic, SensitivityNone, "gateway_parity", "v0.5", "Envoy's upstream time on responses in Envoy mode; gateway-parity checks ignore its value."},
	{EnvoyOverloaded, ToClient, SurfacePublic, SensitivityNone, "upstream", "v0.5", "`true` on the 503 a full circuit breaker returns, as Envoy sets it."},

	// Intermediate decision details.
	{VSRSelectedCategory, ToClient, SurfaceDebug, SensitivityNone, "classification", "v0.1", "Domain category the classifier chose, such as `math`."},
	{VSRSelectedReasoning, ToClient, SurfaceDebug, SensitivityNone, "classification", "v0.1", "Reasoning mode chosen for the request: `on` or `off`."},
	{VSRSelectedModality, ToClient, SurfaceDebug, SensitivityNone, "classification", "v0.2", "Modality result and optional method, such as `AR;classifier`."},
	{VSRDecisionRanking, ToClient, SurfaceDebug, SensitivityNone, "routing", "v0.5", "The matched decision the selected one beat, and the comparison that settled it."},
	{VSRSessionPhase, ToClient, SurfaceDebug, SensitivityNone, "router_learning", "v0.3", "Protection phase: `user_turn`, `tool_loop` or `provider_state`."},
	{VSRLearningMethods, ToClient, SurfaceDebug, SensitivityNone, "router_learning", "v0.4", "Router Learning methods the learning headers summarize, such as `adaptation,protection`."},
	{VSRLearningActions, ToClient, SurfaceDebug, SensitivityNone, "router_learning", "v0.4", "Learning action per method, such as `adaptation=propose_switch`."},
	{VSRLearningScopes, ToClient, SurfaceDebug, SensitivityNone, "router_learning", "v0.4", "Identity scope per method, such as `protection=conversation`."},
	{VSRLearningReasons, ToClient, SurfaceDebug, SensitivityNone, "router_learning", "v0.4", "Reason per method, such as `adaptation=sampled_win`."},
	{VSRInjectedSystemPrompt, ToClient, SurfaceDebug, SensitivityNone, "system_prompt", "v0.1", "Whether a `system_prompt` plugin injected text into the request."},
	{VSRPromptCacheAction, ToClient, SurfaceDebug, SensitivityNone, "prompt_cache", "v0.4", "Outcome of the `prompt_cache` plugin: `inserted`, `preserved`, `skipped` or `rejected`."},
	{VSRPromptCacheReason, ToClient, SurfaceDebug, SensitivityNone, "prompt_cache", "v0.4", "Reason for an outcome other than `inserted`: `caller_markers`, `no_eligible_target` or `unsupported_target`."},
	{VSRPromptCacheInserted, ToClient, SurfaceDebug, SensitivityNone, "prompt_cache", "v0.4", "Cache markers the Router inserted, at most `2`."},
	{VSRPromptCachePreserved, ToClient, SurfaceDebug, SensitivityNone, "prompt_cache", "v0.4", "Cache markers the caller supplied."},
	{VSRCacheSimilarity, ToClient, SurfaceDebug, SensitivityNone, "response_cache", "v0.3", "Similarity score of the response-cache lookup."},
	{VSRToolsStrategy, ToClient, SurfaceDebug, SensitivityNone, "tools", "v0.3", "Retriever strategy of semantic tool selection."},
	{VSRToolsConfidence, ToClient, SurfaceDebug, SensitivityNone, "tools", "v0.3", "Highest similarity score among the selected tools."},
	{VSRToolsLatencyMs, ToClient, SurfaceDebug, SensitivityNone, "tools", "v0.3", "Tool-selection retriever latency, in milliseconds."},
	{VSRRetentionDrop, ToClient, SurfaceDebug, SensitivityNone, "retention", "v0.4", "Whether the matched decision's retention directive keeps the response out of the response cache."},
	{VSRRetentionTTLTurns, ToClient, SurfaceDebug, SensitivityNone, "retention", "v0.4", "Retention lifetime the directive sets, in conversation turns."},
	{VSRRetentionKeepCurrentModel, ToClient, SurfaceDebug, SensitivityNone, "retention", "v0.4", "Whether the directive asks later routing to keep the current model."},
	{VSRRetentionPreferPrefix, ToClient, SurfaceDebug, SensitivityNone, "retention", "v0.4", "Whether the directive prefers prefix retention."},

	// Matched signals.
	{VSRMatchedKeywords, ToClient, SurfaceDebug, SensitivityNone, "signals", "v0.1", "Matched `keyword` rules, comma-separated."},
	{VSRMatchedEmbeddings, ToClient, SurfaceDebug, SensitivityNone, "signals", "v0.1", "Matched `embedding` rules."},
	{VSRMatchedDomains, ToClient, SurfaceDebug, SensitivityNone, "signals", "v0.1", "Matched `domain` rules."},
	{VSRMatchedFactCheck, ToClient, SurfaceDebug, SensitivityNone, "signals", "v0.1", "Matched `fact_check` rules."},
	{VSRMatchedUserFeedback, ToClient, SurfaceDebug, SensitivityNone, "signals", "v0.1", "Matched `user_feedback` rules."},
	{VSRMatchedReask, ToClient, SurfaceDebug, SensitivityNone, "signals", "v0.3", "Matched `reask` rules."},
	{VSRMatchedPreference, ToClient, SurfaceDebug, SensitivityNone, "signals", "v0.1", "Matched `preference` rules."},
	{VSRMatchedLanguage, ToClient, SurfaceDebug, SensitivityNone, "signals", "v0.2", "Matched `language` rules."},
	{VSRMatchedAction, ToClient, SurfaceDebug, SensitivityNone, "signals", "v0.5", "Matched `action` rule."},
	{VSRMatchedContext, ToClient, SurfaceDebug, SensitivityNone, "signals", "v0.2", "Matched `context` rules."},
	{VSRContextTokenCount, ToClient, SurfaceDebug, SensitivityNone, "signals", "v0.2", "Token count the `context` signal measured."},
	{VSRMatchedStructure, ToClient, SurfaceDebug, SensitivityNone, "signals", "v0.3", "Matched `structure` rules."},
	{VSRMatchedComplexity, ToClient, SurfaceDebug, SensitivityNone, "signals", "v0.2", "Matched `complexity` rules with their levels, such as `code_complexity:hard`."},
	{VSRMatchedModality, ToClient, SurfaceDebug, SensitivityNone, "signals", "v0.3", "Matched `modality` rules."},
	{VSRMatchedAuthz, ToClient, SurfaceDebug, SensitivityNone, "signals", "v0.2", "Matched `authz` rules."},
	{VSRMatchedJailbreak, ToClient, SurfaceDebug, SensitivityNone, "signals", "v0.2", "Matched `jailbreak` rules, of the request and of the response."},
	{VSRMatchedSafety, ToClient, SurfaceDebug, SensitivityNone, "signals", "v0.4", "Matched content `safety` rules."},
	{VSRMatchedHallucination, ToClient, SurfaceDebug, SensitivityNone, "signals", "v0.4", "Matched `hallucination` rules, once the answer is checked."},
	{VSRMatchedPII, ToClient, SurfaceDebug, SensitivityNone, "signals", "v0.2", "Matched `pii` rules."},
	{VSRMatchedKB, ToClient, SurfaceDebug, SensitivityNone, "signals", "v0.3", "Matched `kb` rules."},
	{VSRMatchedConversation, ToClient, SurfaceDebug, SensitivityNone, "signals", "v0.3", "Matched `conversation` rules."},
	{VSRMatchedEvent, ToClient, SurfaceDebug, SensitivityNone, "signals", "v0.3", "Matched `event` rules."},
	{VSRMatchedInputModality, ToClient, SurfaceDebug, SensitivityNone, "signals", "v0.4", "Matched `input_modality` rules."},
	{VSRMatchedDecisionModel, ToClient, SurfaceDebug, SensitivityNone, "signals", "v0.5", "Matched decision-model signals: rule names, `rule:choice` for choices and `rule:label` for labels."},
	{VSRMatchedProjection, ToClient, SurfaceDebug, SensitivityNone, "signals", "v0.3", "Matched projection outputs."},

	// The Looper's execution trace.
	{VSRLooperModel, ToClient, SurfaceDebug, SensitivityNone, "looper", "v0.2", "Final model of the Looper."},
	{VSRLooperModelsUsed, ToClient, SurfaceDebug, SensitivityNone, "looper", "v0.2", "Models the Looper called, comma-separated."},
	{VSRLooperIterations, ToClient, SurfaceDebug, SensitivityNone, "looper", "v0.2", "Model calls the Looper made."},
	{VSRLooperAlgorithm, ToClient, SurfaceDebug, SensitivityNone, "looper", "v0.2", "Looper algorithm, such as `confidence`."},
	{VSRLooperLatencyMs, ToClient, SurfaceDebug, SensitivityNone, "looper", "v0.4", "Looper wall-clock latency, in milliseconds."},
	{VSRLooperPromptTokens, ToClient, SurfaceDebug, SensitivityNone, "looper", "v0.4", "Prompt tokens across the Looper's calls."},
	{VSRLooperCompletionTokens, ToClient, SurfaceDebug, SensitivityNone, "looper", "v0.4", "Completion tokens across the Looper's calls."},
	{VSRLooperTotalTokens, ToClient, SurfaceDebug, SensitivityNone, "looper", "v0.4", "Total tokens across the Looper's calls."},

	// Client controls.
	{XSessionID, FromClient | ToProvider, SurfaceControl, SensitivityNone, "session", "v0.3", "Client session id; Router Learning `scope: session` protects it, and providers receive it."},
	{XConversationID, FromClient | ToProvider, SurfaceControl, SensitivityNone, "router_learning", "v0.4", "Client conversation id; Router Learning `scope: conversation` protects it, and providers receive it."},
	{XClaudeCodeSessionID, FromClient | ToProvider, SurfaceControl, SensitivityNone, "session", "v0.3", "Conversation id Claude Code sends; `x-session-id` takes precedence, and providers receive it."},
	{DisableRouterMemory, FromClient, SurfaceControl, SensitivityNone, "memory", "v0.3", "`true` keeps Router-managed memory out of the request."},
	{VSRSkipProcessing, FromClient, SurfaceControl, SensitivityNone, "skip_processing", "v0.3", "`true` opts the request out of Router processing when `global.router.skip_processing.enabled` is set."},
	{VSRDebug, FromClient, SurfaceControl, SensitivityNone, "debug_headers", "v0.4", "`true` asks for the debug headers on the response; honored only for a caller `global.router.headers.debug` allows."},
	{VSRCacheControl, FromClient, SurfaceControl, SensitivityNone, "response_cache", "v0.4", "Response-cache directives for the request, when the plugin's `request_controls` allow them."},
	{VSRCompressionControl, FromClient, SurfaceControl, SensitivityNone, "context_compression", "v0.4", "Context-compression directives for the request, when the plugin's `request_controls` allow them."},
	{VSRProtectionRelease, FromClient, SurfaceControl, SensitivityNone, "router_learning", "v0.5", "Any non-empty value asks router-learning protection for a fresh decision on the request's protection key; `global.router.learning.protection.release.header` can rename it."},
	{SRBenchExpectedConfigHash, FromClient, SurfaceControl, SensitivityNone, "benchmark", "v0.4", "Configuration hash an evaluation request requires; another active configuration fails it with 412."},
	{SRBenchMaxInferenceCalls, FromClient, SurfaceControl, SensitivityNone, "benchmark", "v0.4", "Provider-call limit of an evaluation request; only `1` is supported."},

	// Routing and transport internals.
	{SelectedModel, FromClient | ToProvider, SurfaceInternal, SensitivityNone, "routing", "v0.1", "Route key: the model whose route the gateway sends the provider request on; the Router replaces or drops a client's value."},
	{VSRUpstreamHost, FromProvider, SurfaceInternal, SensitivityNone, "kv_transfer", "v0.5", "Endpoint Envoy reports on the provider response for KV registry writes; removed before the client."},
	{VSRKVSourcePod, ToProvider, SurfaceInternal, SensitivityNone, "kv_transfer", "v0.4", "Address of the pod that holds the source model's KV cache."},
	{VSRKVCacheID, ToProvider, SurfaceInternal, SensitivityNone, "kv_transfer", "v0.4", "Opaque id of the source KV block."},
	{VSRKVMapperID, ToProvider, SurfaceInternal, SensitivityNone, "kv_transfer", "v0.4", "Mapper artifact for the source and target model pair."},
	{VSRKVTransferStatus, FromProvider, SurfaceInternal, SensitivityNone, "kv_transfer", "v0.4", "KV reuse outcome the target pod reports: `applied`, `fallback_reprefill` or `unsupported`."},
	{VSRSystemOneBackend, FromClient | ToProvider, SurfaceInternal, SensitivityNone, "systemone", "v0.5", "Marks a System One request one Router forwards to another, which refuses to forward it again."},
	{VSROutcomeSource, FromClient, SurfaceInternal, SensitivityNone, "router_learning", "v0.4", "Outcome provenance a trusted control plane attests to the management API."},
	{VSRLooperRequest, FromClient, SurfaceInternal, SensitivityNone, "looper", "v0.2", "Retired Looper hop header; the Router drops it from every request."},
	{VSRLooperIteration, FromClient, SurfaceInternal, SensitivityNone, "looper", "v0.2", "Retired Looper hop header; the Router drops it from every request."},
	{VSRLooperDecision, FromClient, SurfaceInternal, SensitivityNone, "looper", "v0.2", "Retired Looper hop header; the Router drops it from every request."},
	{VSRFusionDepth, FromClient, SurfaceInternal, SensitivityNone, "looper", "v0.4", "Retired Looper hop header; the Router drops it from every request."},
	{XForwardedFor, FromClient | ToProvider, SurfaceInternal, SensitivityNone, "native_gateway", "v0.5", "Client address chain; the native gateway's access log records it, and providers receive it."},
	{XForwardedProto, FromClient | ToProvider, SurfaceInternal, SensitivityNone, "native_gateway", "v0.5", "Scheme of the client connection; the native gateway sets it when absent, as Envoy does."},
	{EnvoyUpstreamRqTimeoutMs, FromClient | ToProvider, SurfaceInternal, SensitivityNone, "reliability", "v0.5", "Route timeout the Router sets from the decision's `reliability` block."},
	{EnvoyUpstreamRqPerTryTimeoutMs, FromClient | ToProvider, SurfaceInternal, SensitivityNone, "reliability", "v0.5", "Per-try timeout the Router sets from the decision's `reliability` block."},
	{EnvoyMaxRetries, FromClient | ToProvider, SurfaceInternal, SensitivityNone, "reliability", "v0.4", "Retry count the Router sets from the decision's `reliability` block, and `0` on evaluation requests."},
	{EnvoyRetryOn, FromClient | ToProvider, SurfaceInternal, SensitivityNone, "reliability", "v0.5", "Retry conditions the Router sets from the decision's `reliability` block."},
	{EnvoyRetriableStatusCodes, FromClient | ToProvider, SurfaceInternal, SensitivityNone, "reliability", "v0.5", "Retriable status codes the Router sets from the decision's `reliability` block."},
	{EnvoyHedgeOnPerTryTimeout, FromClient | ToProvider, SurfaceInternal, SensitivityNone, "benchmark", "v0.4", "`false` on evaluation requests, so a per-try timeout never hedges."},
	{EnvoyInternal, FromClient, SurfaceInternal, SensitivityNone, "ingress", "v0.5", "Envoy's internal-request marker; the edge drops it from client requests."},
	{EnvoyRetriableHeaderNames, FromClient, SurfaceInternal, SensitivityNone, "ingress", "v0.5", "Envoy retry control; the edge drops it from client requests."},
	{EnvoyRetryGRPCOn, FromClient, SurfaceInternal, SensitivityNone, "ingress", "v0.5", "Envoy retry control; the edge drops it from client requests."},
	{EnvoyUpstreamAltStatName, FromClient, SurfaceInternal, SensitivityNone, "ingress", "v0.5", "Envoy stats control; the edge drops it from client requests."},
	{EnvoyUpstreamRqTimeoutAltResponse, FromClient, SurfaceInternal, SensitivityNone, "ingress", "v0.5", "Envoy timeout control; the edge drops it from client requests."},
	{EnvoyExpectedRqTimeoutMs, FromClient, SurfaceInternal, SensitivityNone, "ingress", "v0.5", "Envoy timeout control; the edge drops it from client requests."},
	{EnvoyForceTrace, FromClient, SurfaceInternal, SensitivityNone, "ingress", "v0.5", "Envoy tracing control; the edge drops it from client requests."},
	{EnvoyIPTags, FromClient, SurfaceInternal, SensitivityNone, "ingress", "v0.5", "Envoy IP tagging control; the edge drops it from client requests."},
	{EnvoyOriginalURL, FromClient, SurfaceInternal, SensitivityNone, "ingress", "v0.5", "Envoy redirect control; the edge drops it from client requests."},
	{EnvoyRateLimited, FromProvider, SurfaceInternal, SensitivityNone, "upstream", "v0.5", "Marks a provider response an Envoy rate limit rejected; the native upstream can retry it."},

	// Caller identity, which only the authenticator in front of the Router
	// sets.
	{AuthzUserID, FromClient, SurfaceInternal, SensitivityIdentity, "authz", "v0.2", "Authenticated user id; `identity.user_id_header` names another header."},
	{AuthzUserGroups, FromClient, SurfaceInternal, SensitivityIdentity, "authz", "v0.2", "Authenticated user's groups, comma-separated; `identity.user_groups_header` names another header."},
	{AuthzTeamID, FromClient, SurfaceInternal, SensitivityIdentity, "authz", "v0.4", "Authenticated team id; partitions the response cache."},
	{AuthzTenantID, FromClient, SurfaceInternal, SensitivityIdentity, "authz", "v0.4", "Authenticated tenant id; partitions the response cache."},
	{VSROutcomePrincipal, FromClient, SurfaceInternal, SensitivityIdentity, "router_learning", "v0.4", "Opaque attested identity that isolates outcome-ingest rate limits; never persisted."},

	// Credentials. Per-user keys reach a provider only as its credential.
	{UserOpenAIKey, FromClient, SurfaceInternal, SensitivitySecret, "authz", "v0.2", "Per-user OpenAI key the authenticator injects."},
	{UserAnthropicKey, FromClient, SurfaceInternal, SensitivitySecret, "authz", "v0.2", "Per-user Anthropic key the authenticator injects."},
	{UserAzureOpenAIKey, FromClient, SurfaceInternal, SensitivitySecret, "authz", "v0.2", "Per-user Azure OpenAI key the authenticator injects."},
	{UserBedrockKey, FromClient, SurfaceInternal, SensitivitySecret, "authz", "v0.2", "Per-user AWS Bedrock token the authenticator injects."},
	{UserGeminiKey, FromClient, SurfaceInternal, SensitivitySecret, "authz", "v0.2", "Per-user Google Gemini key the authenticator injects."},
	{UserVertexAIKey, FromClient, SurfaceInternal, SensitivitySecret, "authz", "v0.2", "Per-user Vertex AI token the authenticator injects."},
	{UserMiniMaxKey, FromClient, SurfaceInternal, SensitivitySecret, "authz", "v0.3", "Per-user MiniMax key the authenticator injects."},
	{UserCloudflareWorkersAIKey, FromClient, SurfaceInternal, SensitivitySecret, "authz", "v0.4", "Per-user Cloudflare Workers AI token the authenticator injects."},
	{VSRGatewayCredential, FromClient, SurfaceInternal, SensitivitySecret, "authz", "v0.5", "Dedicated credential the authenticator attaches for a trusted upstream gateway; only a backend marked `trusted_gateway` receives its value, under its own auth header, and `gateway_credential.header` names another header."},
	{VSRInternalAuth, FromClient, SurfaceInternal, SensitivitySecret, "looper", "v0.4", "In-process request authentication; the Router drops it from every request."},
	{VSRLooperSecret, FromClient, SurfaceInternal, SensitivitySecret, "looper", "v0.3", "Retired Looper hop secret; the Router drops it from every request."},
	{Authorization, FromClient | ToProvider, SurfaceInternal, SensitivitySecret, "credentials", "v0.1", "Bearer credential: a client's key for the Router, and the provider credential the Router sets."},
	{APIKey, FromClient | ToProvider, SurfaceInternal, SensitivitySecret, "credentials", "v0.2", "Azure OpenAI key: a client's authenticates to the Router and is removed, and the Router sets the provider's."},
	{XAPIKey, ToProvider, SurfaceInternal, SensitivitySecret, "credentials", "v0.2", "Provider credential of Anthropic-style APIs."},
	{XGoogAPIKey, ToProvider, SurfaceInternal, SensitivitySecret, "credentials", "v0.4", "Provider credential of Google APIs."},
	{APISubscriptionKey, ToProvider, SurfaceInternal, SensitivitySecret, "credentials", "v0.4", "Provider credential a catalog provider names."},
	{OcpApimSubscriptionKey, ToProvider, SurfaceInternal, SensitivitySecret, "credentials", "v0.5", "Azure API Management credential, which a model connection sets through `auth`."},
}
