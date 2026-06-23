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

	// XSessionID allows clients to pin a stable session identifier for Chat Completions
	// when router-derived session IDs are insufficient.
	XSessionID = "x-session-id"

	// XClaudeCodeSessionID is the per-conversation UUID the Claude Code CLI
	// emits on every /v1/messages request belonging to the same chat thread.
	// The router mirrors this into RequestContext.SessionID with priority
	// below x-session-id (operator/SDK override) but above metadata.user_id
	// and the message-fingerprint fallbacks. See docs/sessions.md for the
	// full priority order.
	XClaudeCodeSessionID = "x-claude-code-session-id"

	// DisableRouterMemory allows clients to opt-out of router-managed memory injection.
	// This prevents "silent double injection" when applications use SDK-managed memory
	// systems like Mem0, LangMem, LangGraph, or OpenClaw.
	// Value: "true" to disable router memory, any other value or absence enables it.
	// Example use case: App with Mem0 sends this header to prevent duplicate memory injection.
	DisableRouterMemory = "x-disable-router-memory"

	// SelectedModel indicates the model that was selected by the router for processing.
	// This header is set during the routing decision phase.
	SelectedModel = "x-selected-model"

	// VSRSkipProcessing opts the request out of all router processing when
	// global.router.skip_processing.enabled is true. Value: "true" (case-insensitive).
	// See https://github.com/vllm-project/semantic-router/issues/1808.
	VSRSkipProcessing = "x-vsr-skip-processing"

	// VSRDebug opts a request into verbose/debug response headers. Value: "true"
	// (case-insensitive). When set, headers that the v0.4 contract otherwise
	// omits or demotes to replay are emitted inline for that request — the
	// debug/replay-mode trigger for the v0.4 header surface. See issue #2216.
	VSRDebug = "x-vsr-debug"
)

// VSR Decision Tracking Headers
// These headers are added to successful responses (HTTP 200-299) to track
// Vector Semantic Router decision-making information for debugging and monitoring.
// Headers are only added when the request is successful and did not hit the cache.
const (
	// VSRSelectedCategory indicates the category selected by VSR during domain classification.
	// This comes from the domain classifier (MMLU categories).
	// Example values: "math", "business", "biology", "computer science"
	VSRSelectedCategory = "x-vsr-selected-category"

	// VSRSelectedDecision indicates the decision selected by VSR during decision evaluation.
	// This is the final routing decision made by the DecisionEngine.
	// Example values: "math_decision", "business_decision", "thinking_decision"
	VSRSelectedDecision = "x-vsr-selected-decision"

	// VSRSelectedConfidence indicates the confidence score of the selected decision.
	// Value: decimal between 0.0 and 1.0 (e.g., "0.75")
	VSRSelectedConfidence = "x-vsr-selected-confidence"

	// VSRSelectedReasoning indicates whether reasoning mode was determined to be used.
	// Values: "on" (reasoning enabled) or "off" (reasoning disabled)
	VSRSelectedReasoning = "x-vsr-selected-reasoning"

	// VSRSelectedModality indicates the response modality determined by the modality router.
	// Values: "AR" (text-only), "DIFFUSION" (image generation), "BOTH" (text + image)
	VSRSelectedModality = "x-vsr-selected-modality"

	// VSRSelectedModel indicates the model selected by VSR for processing the request.
	// Example values: "deepseek-v31", "phi4", "gpt-4"
	VSRSelectedModel = "x-vsr-selected-model"

	// VSRSessionPhase indicates the Router Learning protection phase.
	// Example values: "user_turn", "tool_loop", "provider_state"
	VSRSessionPhase = "x-vsr-session-phase"

	// VSRLearningMethods names Router Learning methods summarized by the
	// companion learning headers. Full diagnostics live in Router Replay.
	// Example: "adaptation" or "adaptation,protection"
	VSRLearningMethods = "x-vsr-learning-methods"

	// VSRLearningActions contains method-keyed compact learning decisions.
	// Example: "adaptation=propose_switch,protection=allow_switch"
	VSRLearningActions = "x-vsr-learning-actions"

	// VSRLearningScopes contains method-keyed identity scopes.
	// Example: "protection=conversation"
	VSRLearningScopes = "x-vsr-learning-scopes"

	// VSRLearningReasons contains method-keyed machine-readable reasons.
	// Example: "adaptation=sampled_win,protection=switch_allowed"
	VSRLearningReasons = "x-vsr-learning-reasons"

	// VSRInjectedSystemPrompt indicates whether a system prompt was injected into the request.
	// Values: "true" or "false"
	VSRInjectedSystemPrompt = "x-vsr-injected-system-prompt"

	// --- v0.4 keystone response-contract headers (issue #2203) ---
	// These two headers are emitted on every VSR-processed response and form
	// the foundation of the v0.4 header contract: schema-version stamps the
	// contract revision and response-path names how the response was produced.
	// Everything else in the contract keys off response-path.

	// VSRSchemaVersion stamps the response-header contract revision so clients
	// know which contract they are parsing. Value: SchemaVersionValue.
	VSRSchemaVersion = "x-vsr-schema-version"

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

	// SchemaVersionValue is the current response-header contract revision
	// emitted in VSRSchemaVersion. v0.4 is contract revision "2".
	SchemaVersionValue = "2"

	// VSRCacheHit indicates that the response was served from cache.
	// Value: "true"
	VSRCacheHit = "x-vsr-cache-hit"

	// Retention directive headers expose the matched decision's EMIT retention
	// block to the inference pool / operators (issue #2009). They are emitted
	// only on successful, non-cache-hit responses, and only for the fields the
	// directive explicitly set (tri-state). They let the pool observe the
	// router's KV/cache retention intent at the wire; the router itself also
	// consumes drop (cache-write skip) and ttl_turns (per-entry cache TTL).
	VSRRetentionDrop             = "x-vsr-retention-drop"
	VSRRetentionTTLTurns         = "x-vsr-retention-ttl-turns"
	VSRRetentionKeepCurrentModel = "x-vsr-retention-keep-current-model"
	VSRRetentionPreferPrefix     = "x-vsr-retention-prefer-prefix"

	// RouterReplayID carries the identifier for a captured replay record.
	// Value: opaque replay token
	RouterReplayID = "x-vsr-replay-id"

	// VSRToolsStrategy is the name of the retriever strategy that was used
	// during semantic tool selection for this request.
	VSRToolsStrategy = "x-vsr-tools-strategy"

	// VSRToolsConfidence is the similarity score (0–1) of the highest-ranked
	// tool returned by the retriever.  Emitted only when tool selection runs.
	VSRToolsConfidence = "x-vsr-tools-confidence"

	// VSRToolsLatencyMs is the wall-clock time in milliseconds spent inside
	// the retriever for this request.
	VSRToolsLatencyMs = "x-vsr-tools-latency-ms"
)

// VSR Signal Tracking Headers
// These headers track which signals were matched during request evaluation.
// They provide visibility into the signal-driven decision process.
const (
	// VSRMatchedKeywords contains comma-separated list of matched keyword rule names.
	// Example: "code_keywords,urgent_keywords"
	VSRMatchedKeywords = "x-vsr-matched-keywords"

	// VSRMatchedEmbeddings contains comma-separated list of matched embedding rule names.
	// Example: "code_debug,technical_help"
	VSRMatchedEmbeddings = "x-vsr-matched-embeddings"

	// VSRMatchedDomains contains comma-separated list of matched domain rule names.
	// Example: "computer science,mathematics"
	VSRMatchedDomains = "x-vsr-matched-domains"

	// VSRMatchedFactCheck contains the fact-check signal result.
	// Values: "needs_fact_check" or "no_fact_check_needed"
	VSRMatchedFactCheck = "x-vsr-matched-fact-check"

	// VSRMatchedUserFeedback contains comma-separated list of matched user feedback signals.
	// Example: "need_clarification,wrong_answer"
	VSRMatchedUserFeedback = "x-vsr-matched-user-feedback"

	// VSRMatchedReask contains comma-separated list of matched repeated-question dissatisfaction signals.
	// Example: "likely_dissatisfied,persistently_dissatisfied"
	VSRMatchedReask = "x-vsr-matched-reask"

	// VSRMatchedPreference contains comma-separated list of matched preference signals.
	// Example: "creative_writing,technical_analysis"
	VSRMatchedPreference = "x-vsr-matched-preference"

	// VSRMatchedLanguage contains comma-separated list of matched language signals.
	// Example: "en,zh,es"
	VSRMatchedLanguage = "x-vsr-matched-language"

	// VSRMatchedContext contains comma-separated list of matched context rule names.
	// Example: "low_token_count,high_token_count"
	VSRMatchedContext = "x-vsr-matched-context"

	// VSRContextTokenCount contains the actual token count for the request.
	// Example: "1500"
	//nolint:gosec
	VSRContextTokenCount = "x-vsr-context-token-count"

	// VSRMatchedStructure contains comma-separated list of matched structure rule names.
	// Example: "many_questions,numbered_steps"
	VSRMatchedStructure = "x-vsr-matched-structure"

	// VSRMatchedComplexity contains comma-separated list of matched complexity rules with difficulty levels.
	// Example: "code_complexity:hard,math_complexity:easy"
	VSRMatchedComplexity = "x-vsr-matched-complexity"

	// VSRMatchedModality contains comma-separated list of matched modality signals.
	// Example: "AR,DIFFUSION"
	VSRMatchedModality = "x-vsr-matched-modality"

	// VSRMatchedAuthz contains comma-separated list of matched authz rule names.
	// Example: "premium_tier,admin_tier"
	VSRMatchedAuthz = "x-vsr-matched-authz"

	// VSRMatchedJailbreak contains comma-separated list of matched jailbreak rule names.
	// Example: "jailbreak_detected,strict_jailbreak"
	VSRMatchedJailbreak = "x-vsr-matched-jailbreak"

	// VSRMatchedPII contains comma-separated list of matched PII rule names.
	// Example: "pii_strict,pii_moderate"
	VSRMatchedPII = "x-vsr-matched-pii"

	// VSRMatchedKB contains comma-separated list of matched knowledge-base signal names.
	// Example: "privacy_policy,security_containment"
	VSRMatchedKB = "x-vsr-matched-kb"

	// VSRMatchedConversation contains comma-separated list of matched conversation-shape signal names.
	VSRMatchedConversation = "x-vsr-matched-conversation"

	// VSRMatchedEvent contains comma-separated list of matched event signal names.
	// Example: "critical_payment_event,payment_failed"
	VSRMatchedEvent = "x-vsr-matched-event"

	// VSRMatchedProjection contains comma-separated list of matched projection outputs.
	// Example: "balance_medium,verification_required"
	VSRMatchedProjection = "x-vsr-matched-projections"

	// VSRFastResponse indicates that the response was generated by the fast_response plugin
	// without hitting an upstream model.
	// Value: "true"
	VSRFastResponse = "x-vsr-fast-response"
)

// VSR Protocol Markers and Translation Warnings
// These headers are emitted on every non-cache-hit response (including 4xx
// and 5xx) so clients can always tell which translation cell handled the
// call and what was lost during translation.
const (
	// VSRClientProtocol describes the wire format of the inbound request
	// as seen by the router (e.g. "openai", "anthropic"). Defaults to
	// "openai" when no other protocol was detected.
	VSRClientProtocol = "x-vsr-client-protocol"

	// VSRUpstreamProtocol describes the wire format of the outbound
	// request sent to the upstream backend. Defaults to "openai" when no
	// explicit APIFormat was resolved.
	VSRUpstreamProtocol = "x-vsr-upstream-protocol"

	// VSRProtocolWarnings carries a structured, comma-separated list
	// of translation observations emitted by the inbound parser during
	// a lossy translation. Each entry is "severity;reason;field".
	// Absent when no warnings were produced.
	VSRProtocolWarnings = "x-vsr-protocol-warnings"
)

// Response Warnings Header (v0.4)
// VSRResponseWarnings consolidates the response-quality warnings into a single
// comma-separated header on the default response surface (#2204, #2200). The
// per-warning detail (hallucination spans, jailbreak type/confidence,
// fact-check verification context) stays in the replay record, recoverable via
// x-vsr-replay-id. Absent when no warnings were produced.
const (
	VSRResponseWarnings = "x-vsr-response-warnings"

	// Warning codes carried, in order, in the VSRResponseWarnings value.
	ResponseWarningHallucination     = "hallucination"
	ResponseWarningUnverifiedFactual = "unverified_factual"
	ResponseWarningJailbreak         = "response_jailbreak"
)

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
)

// Looper Request Headers
// These headers are added to looper internal requests to identify them
// and allow the extproc to lookup decision configuration and apply plugins.
const (
	// VSRLooperRequest indicates this is an internal looper request.
	// When present, extproc should lookup the decision and execute configured plugins.
	// Value: "true"
	VSRLooperRequest = "x-vsr-looper-request"

	// VSRLooperIteration indicates the current iteration number in the looper loop.
	// Value: "1", "2", "3", etc.
	VSRLooperIteration = "x-vsr-looper-iteration"

	// VSRLooperDecision indicates the decision name for looper internal requests.
	// Used by extproc to lookup decision configuration and apply plugins.
	// Value: decision name (e.g., "remom_low_effort")
	VSRLooperDecision = "x-vsr-looper-decision"

	// VSRFusionDepth marks internal Fusion subrequests to prevent recursive Fusion execution.
	VSRFusionDepth = "x-vsr-fusion-depth"
)

// Looper Response Headers
// These headers are added to responses when looper mode is used.
const (
	// VSRLooperModel indicates the final model used by the looper.
	// Value: model name (e.g., "qwen-max")
	VSRLooperModel = "x-vsr-looper-model"

	// VSRLooperModelsUsed contains the comma-separated list of models that were called.
	// Value: "qwen-flash,qwen-max" (example)
	VSRLooperModelsUsed = "x-vsr-looper-models-used"

	// VSRLooperIterations indicates the total number of model calls made.
	// Value: "2", "3", etc.
	VSRLooperIterations = "x-vsr-looper-iterations"

	// VSRLooperAlgorithm indicates the algorithm used by the looper.
	// Value: "confidence", "ratings", "cost-aware"
	VSRLooperAlgorithm = "x-vsr-looper-algorithm"
)
