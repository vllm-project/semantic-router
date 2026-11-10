package headers

// Package headers provides constants for all custom HTTP headers used in the semantic router.
// All custom headers follow the "x-" prefix convention for non-standard HTTP headers.

// HTTP/2 Pseudo-Headers
// Standard HTTP/2 headers defined in RFC 7540, prefixed with ":".
const (
	// PseudoHeaderPath is the HTTP/2 :path pseudo-header containing the request path and query.
	// Example: "/v1/chat/completions?stream=true"
	PseudoHeaderPath = ":path"
)

// Request Headers
// These headers are used in incoming requests to the semantic router.
const (
	// RequestID is the unique identifier for tracking a request through the system.
	// This header is case-insensitive when read from incoming requests.
	RequestID = "x-request-id"

	// VSRSessionID pins the session a request belongs to. When it is absent,
	// the Router reads global.router.headers.session_id_headers in order.
	VSRSessionID = "x-vsr-session-id"

	// XSessionID and XClaudeCodeSessionID are the client-facing session
	// headers session_id_headers accepts by default; Claude Code sends the
	// latter on every /v1/messages request of a conversation.
	XSessionID           = "x-session-id"
	XClaudeCodeSessionID = "x-claude-code-session-id"

	// XConversationID is the default header Router Learning protection reads a
	// stable conversation identifier from.
	XConversationID = "x-conversation-id"

	// VSRMemory set to MemoryOff keeps Router-managed memory out of the
	// request, for applications whose SDK already manages memory (Mem0,
	// LangMem or LangGraph), so memory is never injected twice.
	VSRMemory = "x-vsr-memory"
	MemoryOff = "off"

	// VSRRouteModel is the route key: the model whose route the gateway sends
	// the provider request on. The Router sets it; a client's value is dropped.
	VSRRouteModel = "x-vsr-route-model"

	// VSRSkipProcessing opts the request out of all router processing when
	// global.router.skip_processing.enabled is true. Value: "true" (case-insensitive).
	// See https://github.com/vllm-project/semantic-router/issues/1808.
	VSRSkipProcessing = "x-vsr-skip-processing"

	// VSRDebug set to "true" (case-insensitive) asks for VSRTrace on the
	// response. The Router honors it only for a caller that
	// global.router.headers.debug allows.
	VSRDebug = "x-vsr-debug"

	// VSRProtectionRelease asks router-learning protection for a fresh
	// decision on the request's protection key. Any non-empty value requests
	// it; global.router.learning.protection.release.header can rename it.
	VSRProtectionRelease = "x-vsr-protection-release"

	// VSRCacheControl is the default header of the response_cache plugin's
	// request controls; request_controls.header names another.
	VSRCacheControl = "x-vsr-cache-control"

	// VSRCompressionControl is the default header of the context plugin's
	// compress request controls; request_controls.header names another.
	VSRCompressionControl = "x-vsr-compression-control"
)

// The public response headers. Every routed response carries the ones that
// apply to it, including cache hits and fast responses.
const (
	// VSRSchemaVersion stamps the response-header contract revision so clients
	// know which contract they are parsing. Value: SchemaVersionValue.
	VSRSchemaVersion = "x-vsr-schema-version"

	// SchemaVersionValue is the current response-header contract revision:
	// the public set, x-vsr-warnings and x-vsr-trace.
	SchemaVersionValue = "3"

	// VSRResponsePath names the final path that produced the response.
	// Value is one of the ResponsePath* constants below.
	VSRResponsePath = "x-vsr-response-path"

	// ResponsePath* are the valid values for VSRResponsePath.
	ResponsePathUpstream        = "upstream"         // forwarded to and answered by the upstream model
	ResponsePathCache           = "cache"            // served from semantic cache, no upstream call
	ResponsePathFastResponse    = "fast_response"    // short-circuited by the fast_response plugin
	ResponsePathLooper          = "looper"           // produced by the agent looper
	ResponsePathImageGeneration = "image_generation" // produced by the image-generation path
	ResponsePathBlocked         = "blocked"          // rejected by a guardrail (e.g. jailbreak/PII)
	ResponsePathRateLimited     = "rate_limited"     // rejected by rate limiting
	ResponsePathError           = "error"            // router-side error response
	ResponsePathFallback        = "fallback"         // produced by upstream error fallback

	// VSRSelectedRecipe identifies the isolated routing profile selected by the
	// inbound virtual model. Concrete backend model requests omit this header.
	VSRSelectedRecipe = "x-vsr-selected-recipe"

	// VSRSelectedDecision is the decision the decision engine selected. It is
	// omitted when no decision matched and the request went to the default model.
	VSRSelectedDecision = "x-vsr-selected-decision"

	// VSRSelectedModel is the model alias the Router selected.
	VSRSelectedModel = "x-vsr-selected-model"

	// RouterReplayID carries the identifier for a captured replay record.
	// Value: opaque replay token
	RouterReplayID = "x-vsr-replay-id"

	// VSRWarnings lists, as an RFC 8941 list, the warnings about the response:
	// the codes of the response_guard rules that annotated the answer, in rule
	// order, then the protocol translation losses. Detail stays in the replay
	// record.
	VSRWarnings = "x-vsr-warnings"

	// VSRCost is the response's usage priced with the served model's configured
	// pricing. Buffered responses only; omitted when the model has no pricing.
	VSRCost = "x-vsr-cost"

	// VSRCostCurrency is the currency of VSRCost. Example: "USD"
	VSRCostCurrency = "x-vsr-cost-currency"
)

// Warning codes in VSRWarnings: a response_guard rule over a hallucination
// signal (unverified_factual when the answer could not be checked), over a
// response-stage jailbreak signal, or over any other signal; a protocol
// translation loss, with its action, reason and field; and the marker of
// warnings left out to keep the header bounded, with their count.
const (
	WarningHallucination     = "hallucination"
	WarningUnverifiedFactual = "unverified_factual"
	WarningJailbreak         = "response_jailbreak"
	WarningGuard             = "response_guard"
	WarningProtocol          = "protocol"
	WarningTruncated         = "truncated"
)

// VSRTrace is the one debug header: an RFC 8941 dictionary of the routing
// details (TraceMembers) for a caller global.router.headers.debug allows.
const VSRTrace = "x-vsr-trace"

// Auth Backend Injected Headers
// These headers are set by the external authorization service (Authorino, Envoy Gateway JWT,
// oauth2-proxy, etc.) after successful user authentication.
// They carry per-user provider API keys and identity for routing.
const (
	// UserOpenAIKey carries the user's OpenAI API key, injected by the auth backend.
	// Used by the ext_proc when routing requests to OpenAI models.
	UserOpenAIKey = "x-user-openai-key"

	// UserAnthropicKey carries the user's Anthropic API key, injected by the auth backend.
	// Used by the ext_proc when routing requests to Anthropic models.
	UserAnthropicKey = "x-user-anthropic-key"

	// UserAzureOpenAIKey carries the user's Azure OpenAI API key, injected by the auth backend.
	UserAzureOpenAIKey = "x-user-azure-openai-key"

	// UserBedrockKey carries the user's AWS Bedrock bearer token, injected by the auth backend.
	UserBedrockKey = "x-user-bedrock-key"

	// UserGeminiKey carries the user's Google Gemini API key, injected by the auth backend.
	UserGeminiKey = "x-user-gemini-key"

	// UserVertexAIKey carries the user's Vertex AI OAuth token, injected by the auth backend.
	UserVertexAIKey = "x-user-vertex-ai-key"

	// UserMiniMaxKey carries the user's MiniMax API key, injected by the auth backend.
	// Used by the ext_proc when routing requests to MiniMax models.
	UserMiniMaxKey = "x-user-minimax-key"

	// UserCloudflareWorkersAIKey carries the user's Cloudflare Workors AI API token,
	// injected by the auth backend. The endpoint is account-scoped, so the account
	// identifier travels in the operator's base URL and only the token is per user.
	UserCloudflareWorkersAIKey = "x-user-cloudflare-workers-ai-key"

	// AuthzUserID is the default header for the authenticated user's identity.
	// Default for Authorino (K8s Secret metadata.name).
	// Override via authz.identity.user_id_header for other backends:
	//   Envoy Gateway JWT: "x-jwt-sub" (from claim_to_headers)
	//   oauth2-proxy:      "x-forwarded-user"
	// Used by the authz signal classifier for user-level routing,
	// and by memory operations for secure per-user isolation.
	AuthzUserID = "x-authz-user-id"

	// AuthzUserGroups is the default header for comma-separated group memberships.
	// Default for Authorino (K8s Secret annotation authz-groups).
	// Override via authz.identity.user_groups_header for other backends:
	//   Envoy Gateway JWT: "x-jwt-groups" (from claim_to_headers)
	//   oauth2-proxy:      "x-forwarded-groups"
	// Used by the authz signal classifier for group-level routing.
	AuthzUserGroups = "x-authz-user-groups"

	// AuthzTeamID and AuthzTenantID are trusted ext_authz outputs used for
	// response-cache partitioning. Client-provided values must be stripped by
	// the gateway before authorization. Like the user and group headers, they
	// never reach a model backend.
	AuthzTeamID   = "x-authz-team-id"
	AuthzTenantID = "x-authz-tenant-id"

	// VSRDelegatedCredential is the default header through which the
	// deployment authenticator hands the Router the requesting user's
	// credential for delegated upstreams (global.services.authz
	// .delegated_credential). It is a trusted authentication output like the
	// identity headers: the gateway strips client-supplied copies, the Router
	// removes it from every upstream request, and only an upstream whose auth
	// is credential: delegated receives its value, under that upstream's auth
	// header. See issue #2286.
	VSRDelegatedCredential = "x-vsr-delegated-credential" // #nosec G101 -- header name, not a credential
)

// Internal Request Authentication
const (
	// VSRInternalAuth authenticates in-process request context that must not
	// be accepted from external callers or forwarded to model backends.
	VSRInternalAuth = "x-vsr-internal-auth"

	// VSROutcomeSource carries server-attested outcome provenance between a
	// trusted control plane and the Router management API. External callers
	// must not be allowed to supply this header through a proxy.
	VSROutcomeSource = "x-vsr-outcome-source"

	// VSROutcomePrincipal carries an opaque, server-attested identity used to
	// isolate outcome-ingest rate limits. It is not persisted with the outcome.
	VSROutcomePrincipal = "x-vsr-outcome-principal"

	// VSRSystemOneBackend marks a System One request one Router forwards to
	// another, so the receiving Router refuses to forward it again. It only
	// restricts the receiving request; it never grants authorization or
	// trusts a caller's identity.
	VSRSystemOneBackend = "x-vsr-systemone-backend"
)

// VSR Cross-Model KV Transfer Request Headers (issue #2976)
// Injected by the KVTransfer Coordinator on upstream dispatch when a model switch
// is eligible for cross-model KV reuse. Consumed by the vLLM KVConnector plugin
// on the target pod.
const (
	// VSRUpstreamHost carries Envoy's %UPSTREAM_HOST% for KV registry writes.
	VSRUpstreamHost = "x-vsr-upstream-host"

	// VSRKVSourcePod is the gRPC address of the pod holding the source model's KV cache.
	// Example: "10.0.1.5:8000"
	VSRKVSourcePod = "x-vsr-kv-source-pod"

	// VSRKVCacheID is the opaque session/cache identifier for the source KV block.
	// Example: "sess-abc123"
	VSRKVCacheID = "x-vsr-kv-cache-id"

	// VSRKVMapperID names the published ridge-mapper artifact for the source→target pair.
	// Example: "qwen3-14b-32b-v1"
	VSRKVMapperID = "x-vsr-kv-mapper-id"
)
