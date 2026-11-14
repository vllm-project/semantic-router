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
func (d Direction) String() string { return strings.Join(d.Labels(), ", ") }

// Labels names the hops in order, such as ["request", "provider request"].
func (d Direction) Labels() []string {
	labels := make([]string, 0, len(directionLabels))
	for _, entry := range directionLabels {
		if d&entry.direction != 0 {
			labels = append(labels, entry.label)
		}
	}
	return labels
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

// ProviderCredentials are the credential headers a provider request carries
// only as the Router sets them from the upstream's auth. A client's copy of
// each is removed before the Router dispatches to a provider it configures.
func ProviderCredentials() []string {
	var names []string
	for _, header := range registry {
		if header.Owner == "credentials" {
			names = append(names, header.Name)
		}
	}
	return names
}

// DroppedFromClients are the Router's own headers no client may supply:
// every registered x-vsr- header that is not a control header, the route
// key, the internal authentication and the KV hints among them. The Router
// never believes a client's value of one, and removes it from the
// provider-bound request before it sets its own. The authenticator's outputs
// are left out: its delegated credential is trusted like the identity headers,
// whose client copies the gateway in front strips.
func DroppedFromClients() []string {
	var names []string
	for _, header := range registry {
		if strings.HasPrefix(header.Name, VSRPrefix) && header.Surface != SurfaceControl && header.Owner != "authz" {
			names = append(names, header.Name)
		}
	}
	return names
}

// RouterOwned reports whether the Router or Envoy owns a header, which a
// decision's header_mutation therefore may not write: every header in the
// x-vsr- and x-envoy- namespaces, and every internal or secret header the
// registry records, the route key and the provider credentials among them.
func RouterOwned(name string) bool {
	normalized := strings.ToLower(strings.TrimSpace(name))
	if strings.HasPrefix(normalized, VSRPrefix) || strings.HasPrefix(normalized, EnvoyPrefix) {
		return true
	}
	header, ok := registryByName[normalized]
	return ok && (header.Surface == SurfaceInternal || header.Sensitivity != SensitivityNone)
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
	{VSRSchemaVersion, ToClient, SurfacePublic, SensitivityNone, "response_contract", "v0.4", "Response-header contract revision; currently `3`."},
	{VSRResponsePath, ToClient, SurfacePublic, SensitivityNone, "response_contract", "v0.4", "How the response was produced: `upstream`, `cache`, `fast_response`, `looper`, `blocked`, `rate_limited`, `error` or `fallback`."},
	{VSRConfigVersion, ToClient, SurfacePublic, SensitivityNone, "config", "v0.5", "Version of the configuration snapshot that served the request."},
	{VSRConfigHash, ToClient, SurfacePublic, SensitivityNone, "config", "v0.4", "SHA-256 of the active runtime document, on routing previews and `GET /api/v1/config`."},
	{VSRWarnings, ToClient, SurfacePublic, SensitivityNone, "response_contract", "v0.5", "Warnings about the response, as an RFC 8941 list: `scan_partial` when a request-stage guard rule read only part of a long input, `guard_overloaded` on the retryable error answered instead of a refusal or a forward that rules unread on an overloaded decision runtime decide, the codes of the `response_guard` rules that annotated the answer, then each `protocol` translation loss with its `action`, `reason` and `field`."},

	// Final routing facts.
	{VSRSelectedRecipe, ToClient, SurfacePublic, SensitivityNone, "routing", "v0.4", "Recipe the effective entrypoint selected; omitted for concrete backend models."},
	{VSRSelectedDecision, ToClient, SurfacePublic, SensitivityNone, "routing", "v0.1", "Decision the decision engine selected; omitted when no decision matched."},
	{VSRSelectedModel, ToClient, SurfacePublic, SensitivityNone, "routing", "v0.1", "Model alias the Router selected."},
	{RouterReplayID, ToClient, SurfacePublic, SensitivityNone, "router_replay", "v0.2", "Router Replay record id, when the record was persisted."},
	{VSRCost, ToClient, SurfacePublic, SensitivityNone, "cost", "v0.4", "Usage priced with the served model's configured pricing; buffered responses only."},
	{VSRCostCurrency, ToClient, SurfacePublic, SensitivityNone, "cost", "v0.4", "Currency of `x-vsr-cost`, from `pricing.currency`."},
	{RateLimitLimit, ToClient, SurfacePublic, SensitivityNone, "rate_limit", "v0.2", "Limit of the rate-limit window that rejected the request."},
	{RateLimitRemaining, ToClient, SurfacePublic, SensitivityNone, "rate_limit", "v0.2", "Requests left in that window."},
	{RateLimitReset, ToClient, SurfacePublic, SensitivityNone, "rate_limit", "v0.2", "Unix time the window resets at."},
	{RequestID, FromClient | ToClient | ToProvider, SurfacePublic, SensitivityNone, "observability", "v0.1", "Request correlation id, forwarded to providers; the native gateway adds one when absent, and the management API echoes it."},
	{XContentTypeOptions, ToClient, SurfacePublic, SensitivityNone, "router_replay", "v0.4", "`nosniff` on Router Replay management responses."},
	{EnvoyUpstreamServiceTime, ToClient, SurfacePublic, SensitivityNone, "gateway_parity", "v0.5", "Envoy's upstream time on responses in Envoy mode; gateway-parity checks ignore its value."},
	{EnvoyOverloaded, ToClient, SurfacePublic, SensitivityNone, "upstream", "v0.5", "`true` on the 503 a full circuit breaker returns, as Envoy sets it."},

	// The routing details.
	{VSRTrace, ToClient, SurfaceDebug, SensitivityNone, "debug_headers", "v0.5", "Routing details as an RFC 8941 dictionary of at most 4 KB, for a caller `global.router.headers.debug` allows."},

	// Client controls.
	{VSRSessionID, FromClient, SurfaceControl, SensitivityNone, "session", "v0.5", "Session id the client pins; without it the Router reads the headers `global.router.headers.session_id_headers` names, in order."},
	{XSessionID, FromClient | ToProvider, SurfaceControl, SensitivityNone, "session", "v0.3", "Client session id `session_id_headers` accepts by default; providers receive it."},
	{XClaudeCodeSessionID, FromClient | ToProvider, SurfaceControl, SensitivityNone, "session", "v0.3", "Conversation id Claude Code sends, which `session_id_headers` accepts by default after `x-session-id`; providers receive it."},
	{XConversationID, FromClient | ToProvider, SurfaceControl, SensitivityNone, "router_learning", "v0.4", "Client conversation id; Router Learning `scope: conversation` protects it, and providers receive it."},
	{VSRMemory, FromClient, SurfaceControl, SensitivityNone, "memory", "v0.5", "`off` keeps Router-managed memory out of the request."},
	{VSRSkipProcessing, FromClient, SurfaceControl, SensitivityNone, "skip_processing", "v0.3", "`true` opts the request out of Router processing when `global.router.skip_processing.enabled` is set."},
	{VSRDebug, FromClient, SurfaceControl, SensitivityNone, "debug_headers", "v0.4", "`true` asks for `x-vsr-trace` on the response; honored only for a caller `global.router.headers.debug` allows."},
	{VSRCacheControl, FromClient, SurfaceControl, SensitivityNone, "response_cache", "v0.4", "Response-cache directives for the request, when the plugin's `request_controls` allow them."},
	{VSRCompressionControl, FromClient, SurfaceControl, SensitivityNone, "context", "v0.4", "Context-compression directives for the request, when the plugin's `request_controls` allow them."},
	{VSRProtectionRelease, FromClient, SurfaceControl, SensitivityNone, "router_learning", "v0.5", "Any non-empty value asks router-learning protection for a fresh decision on the request's protection key; `global.router.learning.protection.release.header` can rename it."},
	{VSRBenchExpectedConfigHash, FromClient, SurfaceControl, SensitivityNone, "benchmark", "v0.5", "Configuration hash an evaluation request requires; another active configuration fails it with 412."},
	{VSRBenchMaxInferenceCalls, FromClient, SurfaceControl, SensitivityNone, "benchmark", "v0.5", "Provider-call limit of an evaluation request; only `1` is supported."},

	// Routing and transport internals.
	{VSRRouteModel, FromClient | ToProvider, SurfaceInternal, SensitivityNone, "routing", "v0.5", "Route key: the model whose route the gateway sends the provider request on; the Router drops a client's value and sets its own."},
	{VSRUpstreamHost, FromProvider, SurfaceInternal, SensitivityNone, "kv_transfer", "v0.5", "Endpoint Envoy reports on the provider response for KV registry writes; removed before the client."},
	{VSRKVSourcePod, ToProvider, SurfaceInternal, SensitivityNone, "kv_transfer", "v0.4", "Address of the pod that holds the source model's KV cache."},
	{VSRKVCacheID, ToProvider, SurfaceInternal, SensitivityNone, "kv_transfer", "v0.4", "Opaque id of the source KV block."},
	{VSRKVMapperID, ToProvider, SurfaceInternal, SensitivityNone, "kv_transfer", "v0.4", "Mapper artifact for the source and target model pair."},
	{VSRSystemOneBackend, FromClient | ToProvider, SurfaceInternal, SensitivityNone, "systemone", "v0.5", "Marks a System One request one Router forwards to another, which refuses to forward it again."},
	{VSROutcomeSource, FromClient, SurfaceInternal, SensitivityNone, "router_learning", "v0.4", "Outcome provenance a trusted control plane attests to the management API."},
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
	{VSRDelegatedCredential, FromClient, SurfaceInternal, SensitivitySecret, "authz", "v0.5", "The requesting user's credential the authenticator attaches; only an upstream with `auth.credential: delegated` receives its value, under its own auth header, and `delegated_credential.header` names another header."},
	{VSRInternalAuth, FromClient, SurfaceInternal, SensitivitySecret, "looper", "v0.4", "In-process request authentication; the Router drops it from every client request."},
	{Authorization, FromClient | ToProvider, SurfaceInternal, SensitivitySecret, "credentials", "v0.1", "Bearer credential: a client's key for the Router, and the provider credential the Router sets."},
	{APIKey, FromClient | ToProvider, SurfaceInternal, SensitivitySecret, "credentials", "v0.2", "Azure OpenAI key: a client's authenticates to the Router and is removed, and the Router sets the provider's."},
	{XAPIKey, ToProvider, SurfaceInternal, SensitivitySecret, "credentials", "v0.2", "Provider credential of Anthropic-style APIs."},
	{XGoogAPIKey, ToProvider, SurfaceInternal, SensitivitySecret, "credentials", "v0.4", "Provider credential of Google APIs."},
	{APISubscriptionKey, ToProvider, SurfaceInternal, SensitivitySecret, "credentials", "v0.4", "Provider credential a catalog provider names."},
	{OcpApimSubscriptionKey, ToProvider, SurfaceInternal, SensitivitySecret, "credentials", "v0.5", "Azure API Management credential, which a model connection sets through `auth`."},
}
