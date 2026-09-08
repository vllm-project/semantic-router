package extproc

import (
	"context"
	"time"

	"go.opentelemetry.io/otel/trace"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/projectiontrace"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ratelimit"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

// EnhancedHallucinationSpan represents a hallucinated span with NLI explanation.
type EnhancedHallucinationSpan struct {
	Text                    string  `json:"text"`
	Start                   int     `json:"start"`
	End                     int     `json:"end"`
	HallucinationConfidence float32 `json:"hallucination_confidence"`
	NLILabel                string  `json:"nli_label"` // ENTAILMENT, NEUTRAL, or CONTRADICTION
	NLIConfidence           float32 `json:"nli_confidence"`
	Severity                int     `json:"severity"`    // 0-4: 0=low, 4=critical
	Explanation             string  `json:"explanation"` // Human-readable explanation
}

// EnhancedHallucinationInfo contains detailed NLI analysis of hallucinations.
type EnhancedHallucinationInfo struct {
	Confidence float32                     `json:"confidence"`
	Spans      []EnhancedHallucinationSpan `json:"spans"`
}

// RequestContext holds the context for processing a request.
type RequestContext struct {
	Headers   map[string]string
	RequestID string
	// IngressBodyBytes records only transport size. Source bytes live in the
	// bounded, ephemeral protocol envelope and are never general-purpose state.
	IngressBodyBytes  int
	RequestModel      string
	RequestQuery      string
	CacheRequestModel string
	// CacheQuery is the semantic-cache lookup key (may include user scope); empty means fall back to RequestQuery.
	CacheQuery                    string
	CacheExactFingerprint         string
	CacheCompatibilityFingerprint string
	CacheIdentity                 cache.CacheIdentity
	CacheSelectedModel            string
	CacheSemanticSafe             bool
	CacheReadBypass               bool
	CacheWriteBypass              bool
	CacheMaxAgeSeconds            *int
	CacheWriteTTLSeconds          *int
	ContextCompressionApplied     bool
	ContextCompressionBefore      int
	ContextCompressionAfter       int
	ContextCompressionMessages    int
	ContextCompressionFormat      string
	ContextCompressionOmitted     int
	ContextCompressionSkipReason  string
	StartTime                     time.Time
	ProcessingStartTime           time.Time
	// Streaming detection
	ExpectStreamingResponse bool // set from request Accept header or stream parameter
	IsStreamingResponse     bool // set from response Content-Type

	// Semi-streaming body handler (non-nil when Envoy sends STREAMED body chunks)
	StreamedBody          *StreamedBodyHandler
	FullDuplexRequestBody bool // true when the data plane negotiated FULL_DUPLEX_STREAMED
	SkipProcessing        bool // true only when the configured opt-out header is valid

	StreamingComplete      bool // True after neutral stream finalization runs once.
	StreamingAborted       bool // True if the neutral stream ended abnormally.
	ProtocolResponseStream *protocolcodec.StreamEngine
	PublicChatUsageFilter  *protocolcodec.ChatUsageStreamFilter
	SemanticStreamState    *semanticResponseStreamState

	// UpstreamStatusCode is the HTTP status the upstream returned, captured at
	// the response-header phase. Zero means the status was never observed for
	// this request (e.g. response headers not processed). The cache-write path
	// reads it to avoid caching non-2xx error bodies (cache poisoning).
	UpstreamStatusCode int

	// TTFT tracking
	TTFTRecorded bool
	TTFTSeconds  float64

	// InflightToken is the handle returned by inflight.Begin when this request
	// was admitted to the in-flight tracker after model selection. Zero means
	// the request was never admitted (rejected pre-selection, cache hit, etc.)
	// and inflight.End on it is a no-op.
	InflightToken uint64

	// Session-aware transition metadata
	SessionID           string  // Derived from ConversationID (Response API) or message hash (Chat Completions)
	TurnIndex           int     // Number of prior turns in this session (0 = first turn)
	PreviousModel       string  // Model used in the immediately preceding turn; empty on first turn
	CacheWarmthEstimate float64 // [0,1] from EstimateCacheProbability; 0.5 = unknown
	SessionIdleSeconds  float64 // Seconds since the last locally observed turn for this session
	SessionIdleKnown    bool    // True when SessionIdleSeconds came from local session observation

	// HistoryTokenCount is the estimated token count of conversation history,
	// excluding the current turn. Source priority: provider usage accumulation
	// (resp.Usage.InputTokens + OutputTokens) > char/4 text fallback.
	HistoryTokenCount int

	// PreviousResponseID is the Response API previous_response_id field.
	// Non-empty value is a strong explicit continuation signal even when
	// HistoryTokenCount is zero (server-side conversation state).
	PreviousResponseID string

	// Routing is the single source of truth for entrypoint resolution. A
	// resolved context with no recipe represents concrete-model passthrough.
	Routing RequestRoutingContext

	// VSR decision tracking
	VSRSelectedCategory             string                                      // The category from domain classification (MMLU category)
	VSRSelectedDecisionName         string                                      // The decision name from DecisionEngine evaluation
	VSRSelectedDecisionConfidence   float64                                     // Confidence score from DecisionEngine evaluation
	VSRReasoningMode                string                                      // "on" or "off" - whether reasoning mode was determined to be used
	VSRSelectedModel                string                                      // The model selected by VSR
	VSRSelectionMethod              string                                      // Model selection algorithm used (e.g., "elo", "static", "router_dc")
	VSRSelectionReasoning           string                                      // Bounded human-readable selector rationale for replay
	VSRFusionQuorum                 *routerreplay.FusionQuorumDiagnostics       // Content-free Fusion panel quorum evidence for replay
	VSRPromptHelperModel            string                                      // Concrete prompt-selector helper model
	VSRPromptHelperPromptTokens     int64                                       // Prompt tokens consumed by the helper
	VSRPromptHelperCompletionTokens int64                                       // Completion tokens consumed by the helper
	VSRPromptHelperTotalTokens      int64                                       // Total helper tokens
	VSRPromptHelperLatencyMs        int64                                       // Helper model round-trip latency
	VSRLearningPolicy               *routerLearningPolicy                       // Primary Router Learning trace
	VSRLearningPolicies             routerLearningPolicies                      // Router Learning traces by component
	VSRLearningProtectionPreflight  *routerreplay.LearningProtectionDiagnostics // Protection preflight trace for replay
	VSRLearningSessionID            string                                      // Router Learning memory key used for this request
	VSRLearningConversationID       string                                      // Client-declared conversation identity used by Router Learning
	VSRCacheHit                     bool                                        // Whether this request hit the cache
	VSRCacheSimilarity              float32                                     // Similarity score from last cache lookup (0 = no lookup performed)
	VSRCacheHitKind                 string
	VSRCacheSource                  string
	VSRCacheEntryAgeSeconds         float64
	VSRCacheTTLSeconds              int
	VSRInjectedSystemPrompt         bool             // Whether a system prompt was injected into the request
	VSRSelectedDecision             *config.Decision // The decision object selected by DecisionEngine (for plugins)
	// VSREligibleModelRefs is the selected decision's model set after applying
	// request contracts. Loopers consume this exact set; broader Router Learning
	// candidate sets must independently apply the same request contracts.
	VSREligibleModelRefs []config.ModelRef

	// ResponsePath records how the final response was produced, surfaced as the
	// v0.4 keystone x-vsr-response-path header (one of the headers.ResponsePath*
	// values). It defaults to "upstream"; immediate-response paths (cache,
	// fast_response, looper, rate-limit, error, image generation) set it
	// explicitly so the emitted path is accurate. See issue #2203.
	ResponsePath string

	// Modality routing classification result (AR/DIFFUSION/BOTH)
	ModalityClassification *ModalityClassificationResult // Set by classifyModality()

	// VSR signal tracking - stores all matched signals for response headers
	VSRMatchedKeywords        []string // Matched keyword rule names
	VSRMatchedEmbeddings      []string // Matched embedding rule names
	VSRMatchedDomains         []string // Matched domain rule names
	VSRMatchedFactCheck       []string // Matched fact-check signals
	VSRMatchedUserFeedback    []string // Matched user feedback signals
	VSRMatchedReask           []string // Matched repeated-question dissatisfaction signals
	VSRMatchedPreference      []string // Matched preference signals
	VSRMatchedLanguage        []string // Matched language signals
	VSRMatchedContext         []string // Matched context rule names (e.g. "low_token_count")
	VSRContextTokenCount      int      // Conservative request-context token estimate used for routing
	VSRContextTextBytes       int      // Actual semantic-text bytes eligible for online text calibration
	VSRContextEquivalentBytes int      // Content-free byte equivalent of the conservative routing floor
	VSRContextHasNonText      bool     // Structured JSON or image reserves make text-only calibration unsafe
	VSRMatchedStructure       []string // Matched structure rule names
	VSRMatchedComplexity      []string // Matched complexity rules with difficulty level (e.g. "code_complexity:hard")
	VSRMatchedModality        []string // Matched modality signals: "AR", "DIFFUSION", or "BOTH"
	VSRMatchedAuthz           []string // Matched authz rule names for user-level routing
	VSRMatchedJailbreak       []string // Matched jailbreak rule names (confidence >= threshold)
	VSRMatchedPII             []string // Matched PII rule names (denied PII types detected)
	VSRMatchedKB              []string // Matched knowledge-base signal names
	VSRMatchedConversation    []string // Matched conversation-shape signal names
	VSRMatchedEvent           []string // Matched event signal names
	VSRMatchedMetadata        []string // Matched untrusted request metadata signal names
	VSRMatchedClassifier      []string // Matched generic classifier signal names
	VSRMatchedInputModality   []string // Matched structural input-modality signal names
	VSRConversationFacts      classification.ConversationFacts
	VSRMatchedProjection      []string // Matched projection mapping outputs
	VSRProjectionScores       map[string]float64
	VSRSignalConfidences      map[string]float64
	VSRSignalValues           map[string]float64
	VSRSignalErrors           map[string]string
	// VSRMatchedResponseJailbreak holds response-direction jailbreak rules that
	// matched. Populated after the model answers, unlike every VSRMatched*
	// above it.
	VSRMatchedResponseJailbreak []string
	// VSRResponseJailbreakType and VSRResponseJailbreakRisk are the evidence the
	// response-stage signal was computed from, kept so the plugin does not have
	// to re-derive them from the per-rule confidences.
	VSRResponseJailbreakType string
	VSRResponseJailbreakRisk float32
	VSRDecisionDiagnostics   decision.EvaluationDiagnostics
	VSRProjectionTrace       *projectiontrace.Trace

	// Hallucination mitigation tracking
	FactCheckNeeded           bool                       // Result of fact-check classification
	FactCheckConfidence       float32                    // Confidence score of fact-check classification
	HasToolsForFactCheck      bool                       // Request has tools that provide context for fact-checking
	ToolResultsContext        string                     // Aggregated tool results for hallucination check
	UserContent               string                     // Stored user content for hallucination detection
	RequestImageURL           string                     // First image URL from user messages (for Tier 1 complexity classification)
	HallucinationDetected     bool                       // Result of hallucination detection
	HallucinationSpans        []string                   // Unsupported spans found in answer (basic mode)
	HallucinationConfidence   float32                    // Confidence score of hallucination detection
	EnhancedHallucinationInfo *EnhancedHallucinationInfo // Detailed NLI info (when use_nli enabled)
	UnverifiedFactualResponse bool                       // True if fact-check needed but no tools to verify against

	// Jailbreak Detection Results (request-level, from signal classification)
	JailbreakDetected   bool    // True if jailbreak was detected in user input
	JailbreakType       string  // Type of jailbreak detected
	JailbreakConfidence float32 // Confidence score of jailbreak detection

	// Response-level Jailbreak Detection Results (from response body scanning)
	ResponseJailbreakDetected   bool    // True if jailbreak content detected in LLM response
	ResponseJailbreakType       string  // Type of jailbreak detected in response
	ResponseJailbreakConfidence float32 // Confidence score of response jailbreak detection

	// PII Detection Results
	PIIDetected bool     // True if PII was detected
	PIIEntities []string // PII entity types detected (e.g., ["EMAIL", "PHONE_NUMBER"])
	PIIBlocked  bool     // True if request was blocked due to PII policy violation

	// Tracing context
	TraceContext context.Context // OpenTelemetry trace context for span propagation
	UpstreamSpan trace.Span      // Span for tracking upstream vLLM request duration

	// ResponseObjectState is present only when optional Responses object
	// persistence participates in this request. Generation never depends on it.
	ResponseObjectState *ResponseObjectState

	// Router replay context
	RouterReplayID           string                           // ID of the router replay session, if applicable
	RouterReplayPluginConfig *config.RouterReplayPluginConfig // Per-decision plugin configuration for router replay
	RouterReplayRecorder     *routerreplay.Recorder           // The recorder instance for this decision

	// Looper context
	LooperRequest   bool // True only for token-authenticated in-process looper requests
	LooperIteration int  // The iteration number if this is a looper request

	// SourceFormat and SemanticRequest are the authoritative public protocol
	// contract and neutral request.
	SourceFormat             llmprotocol.WireFormat
	TargetFormat             llmprotocol.WireFormat
	SemanticRequest          *llmprotocol.Request
	SemanticResponse         *llmprotocol.Response
	ProtocolEnvelope         llmprotocol.Envelope
	ResponseEnvelope         llmprotocol.Envelope
	ProtocolDiagnostics      llmprotocol.Diagnostics
	ImmediateProtocolError   *llmprotocol.ProtocolError
	ImmediateResponseEncoded bool

	// RAG (Retrieval-Augmented Generation) tracking
	RAGRetrievedContext string  // Retrieved context from RAG plugin
	RAGBackend          string  // Backend used for retrieval ("milvus", "external_api", "mcp", "hybrid")
	RAGSimilarityScore  float32 // Best similarity score from retrieval
	RAGRetrievalLatency float64 // Retrieval latency in seconds
	RAGToolCallIDs      map[string]struct{}

	// Memory retrieval tracking
	// Stores formatted memory context to be injected after system prompt
	MemoryContext        string // Formatted memory context (empty if no memories retrieved)
	MemoryBackend        string
	MemoryStatus         string
	MemoryReason         string
	MemoryFallbackReason string
	MemoryFailOpen       bool
	MemoryResultCount    int
	MemoryMessageIndexes map[int]struct{}

	ContextCompressionTargetTokens *int
	ContextCompressionRecoveryKeys []string
	ContextCompressionStrategy     string
	ContextCompressionBudgetMode   string
	ContextCompressionTokenSource  string
	ContextCompressionTrigger      string
	ContextCompressionRevision     string
	ContextCompressionQuality      string
	ContextCompressionFallback     string
	ContextCompressionCostSaved    float64

	// RateLimitCtx is retained for the existing optional data-plane limiter.
	// Protocol codecs never read or mutate admission state.
	RateLimitCtx *ratelimit.Context

	// EmittedRetention is a deep-clone snapshot of the retention directive
	// emitted by the matched Decision. The clone owns its own pointer fields
	// so that any later mutation does not poison the read-only config tree.
	// nil means the matched decision had no EMIT retention block.
	EmittedRetention *config.RetentionDirective
}

// embeddingContext returns the request-scoped context for outbound embedding
// calls, so a client disconnect cancels in-flight remote embedding RPCs.
func (ctx *RequestContext) embeddingContext() context.Context {
	if ctx == nil || ctx.TraceContext == nil {
		return context.Background()
	}
	return ctx.TraceContext
}

// HasPersonalizedContext returns true if the request/response is tainted with user-specific
// context (RAG, memory, PII) that should prevent caching the response.
// The second return value is the canonical reason for observability (metrics/tracing).
//
// Note: VSRInjectedSystemPrompt is intentionally excluded. System prompts are
// per-decision, not per-user — all users hitting the same decision get the same
// system prompt, so the response is not "personalized" in the cross-user-leak sense.
func (ctx *RequestContext) HasPersonalizedContext() (bool, string) {
	if ctx == nil {
		return false, ""
	}
	if ctx.RAGRetrievedContext != "" {
		return true, "rag_context"
	}
	if ctx.MemoryContext != "" {
		return true, "memory_context"
	}
	if ctx.PIIDetected {
		return true, "pii_detected"
	}
	return false, ""
}
