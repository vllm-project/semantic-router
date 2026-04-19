package extproc

import (
	"context"
	"time"

	"go.opentelemetry.io/otel/trace"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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

// ChatCompletionMessage is role plus plain-text content from a Chat Completions request.
type ChatCompletionMessage struct {
	Role    string
	Content string
}

// StreamingToolCallState accumulates tool-call deltas across streaming chunks.
type StreamingToolCallState struct {
	ID        string
	Name      string
	Arguments string
}

// RequestContext holds the context for processing a request.
type RequestContext struct {
	Headers             map[string]string
	RequestID           string
	OriginalRequestBody []byte
	RequestModel        string
	RequestQuery        string
	// CacheQuery is the semantic-cache lookup key (may include user scope); empty means fall back to RequestQuery.
	CacheQuery          string
	StartTime           time.Time
	ProcessingStartTime time.Time

	// Streaming detection
	ExpectStreamingResponse bool // set from request Accept header or stream parameter
	IsStreamingResponse     bool // set from response Content-Type

	// Semi-streaming body handler (non-nil when Envoy sends STREAMED body chunks)
	StreamedBody *StreamedBodyHandler

	// Streaming accumulation for caching
	StreamingChunks    []string                        // Accumulated SSE chunks
	StreamingContent   string                          // Accumulated content from delta.content
	StreamingMetadata  map[string]interface{}          // id, model, created from first chunk
	StreamingToolCalls map[int]*StreamingToolCallState // Accumulated delta.tool_calls keyed by tool index
	StreamingComplete  bool                            // True when [DONE] marker received
	StreamingAborted   bool                            // True if stream ended abnormally (EOF, cancel, timeout)

	// TTFT tracking
	TTFTRecorded bool
	TTFTSeconds  float64

	// Session-aware transition metadata
	SessionID           string  // Derived from ConversationID (Response API) or message hash (Chat Completions)
	TurnIndex           int     // Number of prior turns in this session (0 = first turn)
	PreviousModel       string  // Model used in the immediately preceding turn; empty on first turn
	CacheWarmthEstimate float64 // [0,1] from EstimateCacheProbability; 0.5 = unknown

	// VSR decision tracking
	VSRSelectedCategory           string           // The category from domain classification (MMLU category)
	VSRSelectedDecisionName       string           // The decision name from DecisionEngine evaluation
	VSRSelectedDecisionConfidence float64          // Confidence score from DecisionEngine evaluation
	VSRReasoningMode              string           // "on" or "off" - whether reasoning mode was determined to be used
	VSRSelectedModel              string           // The model selected by VSR
	VSRSelectionMethod            string           // Model selection algorithm used (e.g., "elo", "static", "router_dc")
	VSRCacheHit                   bool             // Whether this request hit the cache
	VSRCacheSimilarity            float32          // Similarity score from last cache lookup (0 = no lookup performed)
	VSRInjectedSystemPrompt       bool             // Whether a system prompt was injected into the request
	VSRSelectedDecision           *config.Decision // The decision object selected by DecisionEngine (for plugins)

	// Modality routing classification result (AR/DIFFUSION/BOTH)
	ModalityClassification *ModalityClassificationResult // Set by classifyModality()

	// VSR signal tracking - stores all matched signals for response headers
	VSRMatchedKeywords     []string // Matched keyword rule names
	VSRMatchedEmbeddings   []string // Matched embedding rule names
	VSRMatchedDomains      []string // Matched domain rule names
	VSRMatchedFactCheck    []string // Matched fact-check signals
	VSRMatchedUserFeedback []string // Matched user feedback signals
	VSRMatchedReask        []string // Matched repeated-question dissatisfaction signals
	VSRMatchedPreference   []string // Matched preference signals
	VSRMatchedLanguage     []string // Matched language signals
	VSRMatchedContext      []string // Matched context rule names (e.g. "low_token_count")
	VSRContextTokenCount   int      // Actual token count for the request
	VSRMatchedStructure    []string // Matched structure rule names
	VSRMatchedComplexity   []string // Matched complexity rules with difficulty level (e.g. "code_complexity:hard")
	VSRMatchedModality     []string // Matched modality signals: "AR", "DIFFUSION", or "BOTH"
	VSRMatchedAuthz        []string // Matched authz rule names for user-level routing
	VSRMatchedJailbreak    []string // Matched jailbreak rule names (confidence >= threshold)
	VSRMatchedPII          []string // Matched PII rule names (denied PII types detected)
	VSRMatchedKB           []string // Matched knowledge-base signal names
	VSRMatchedConversation []string // Matched conversation-shape signal names
	VSRMatchedProjection   []string // Matched projection mapping outputs
	VSRProjectionScores    map[string]float64
	VSRSignalConfidences   map[string]float64
	VSRSignalValues        map[string]float64

	// Endpoint tracking for windowed metrics
	SelectedEndpoint string // The endpoint address selected for this request
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

	// Response API context
	ResponseAPICtx *ResponseAPIContext // Non-nil if this is a Response API request

	// Chat Completions messages (parsed for memory and related flows)
	ChatCompletionMessages []ChatCompletionMessage
	// ChatCompletionRequestBody is the raw chat-completions JSON (dev: extract user from body).
	ChatCompletionRequestBody []byte
	// ChatCompletionUserID is populated in dev builds when parsing the chat-completions body.
	ChatCompletionUserID string

	// Router replay context
	RouterReplayID           string                           // ID of the router replay session, if applicable
	RouterReplayPluginConfig *config.RouterReplayPluginConfig // Per-decision plugin configuration for router replay
	RouterReplayRecorder     *routerreplay.Recorder           // The recorder instance for this decision

	// Looper context
	LooperRequest   bool // True if this request is from looper (internal request, skip plugins)
	LooperIteration int  // The iteration number if this is a looper request

	// External API routing context (for Envoy-routed external API requests)
	// APIFormat indicates the target API format (e.g., "anthropic", "gemini")
	// Empty string means standard OpenAI-compatible backend (no transformation needed)
	APIFormat string

	// RAG (Retrieval-Augmented Generation) tracking
	RAGRetrievedContext string  // Retrieved context from RAG plugin
	RAGBackend          string  // Backend used for retrieval ("milvus", "external_api", "mcp", "hybrid")
	RAGSimilarityScore  float32 // Best similarity score from retrieval
	RAGRetrievalLatency float64 // Retrieval latency in seconds

	// Memory retrieval tracking
	// Stores formatted memory context to be injected after system prompt
	MemoryContext string // Formatted memory context (empty if no memories retrieved)

	// Note: Per-user API keys from ext_authz / Authorino are read directly from
	// ctx.Headers by the CredentialResolver (pkg/authz). No separate fields needed.

	// Rate limit context - stored after Check() for post-response Report()
	RateLimitCtx *ratelimit.Context
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
