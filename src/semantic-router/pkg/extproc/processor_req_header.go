package extproc

import (
	"context"
	"strings"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

// EnhancedHallucinationSpan represents a hallucinated span with NLI explanation
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

// EnhancedHallucinationInfo contains detailed NLI analysis of hallucinations
type EnhancedHallucinationInfo struct {
	Confidence float32                     `json:"confidence"`
	Spans      []EnhancedHallucinationSpan `json:"spans"`
}

// RequestContext holds the context for processing a request
type RequestContext struct {
	Headers             map[string]string
	RequestID           string
	OriginalRequestBody []byte
	RequestModel        string
	RequestQuery        string
	StartTime           time.Time
	ProcessingStartTime time.Time

	// Streaming detection
	ExpectStreamingResponse bool // set from request Accept header or stream parameter
	IsStreamingResponse     bool // set from response Content-Type

	// Streaming accumulation for caching
	StreamingChunks   []string               // Accumulated SSE chunks
	StreamingContent  string                 // Accumulated content from delta.content
	StreamingMetadata map[string]interface{} // id, model, created from first chunk
	StreamingComplete bool                   // True when [DONE] marker received
	StreamingAborted  bool                   // True if stream ended abnormally (EOF, cancel, timeout)

	// TTFT tracking
	TTFTRecorded bool
	TTFTSeconds  float64

	// VSR decision tracking
	VSRSelectedCategory           string           // The category from domain classification (MMLU category)
	VSRSelectedDecisionName       string           // The decision name from DecisionEngine evaluation
	VSRSelectedDecisionConfidence float64          // Confidence score from DecisionEngine evaluation
	VSRReasoningMode              string           // "on" or "off" - whether reasoning mode was determined to be used
	VSRSelectedModel              string           // The model selected by VSR
	VSRCacheHit                   bool             // Whether this request hit the cache
	VSRInjectedSystemPrompt       bool             // Whether a system prompt was injected into the request
	VSRSelectedDecision           *config.Decision // The decision object selected by DecisionEngine (for plugins)

	// VSR signal tracking - stores all matched signals for response headers
	VSRMatchedKeywords     []string // Matched keyword rule names
	VSRMatchedEmbeddings   []string // Matched embedding rule names
	VSRMatchedDomains      []string // Matched domain rule names
	VSRMatchedFactCheck    []string // Matched fact-check signals
	VSRMatchedUserFeedback []string // Matched user feedback signals
	VSRMatchedPreference   []string // Matched preference signals
	VSRMatchedLanguage     []string // Matched language signals
	VSRMatchedLatency      []string // Matched latency signals
	VSRMatchedContext      []string // Matched context rule names (e.g. "low_token_count")
	VSRContextTokenCount   int      // Actual token count for the request

	// Endpoint tracking for windowed metrics
	SelectedEndpoint string // The endpoint address selected for this request
	// Hallucination mitigation tracking
	FactCheckNeeded           bool                       // Result of fact-check classification
	FactCheckConfidence       float32                    // Confidence score of fact-check classification
	HasToolsForFactCheck      bool                       // Request has tools that provide context for fact-checking
	ToolResultsContext        string                     // Aggregated tool results for hallucination check
	UserContent               string                     // Stored user content for hallucination detection
	HallucinationDetected     bool                       // Result of hallucination detection
	HallucinationSpans        []string                   // Unsupported spans found in answer (basic mode)
	HallucinationConfidence   float32                    // Confidence score of hallucination detection
	EnhancedHallucinationInfo *EnhancedHallucinationInfo // Detailed NLI info (when use_nli enabled)
	UnverifiedFactualResponse bool                       // True if fact-check needed but no tools to verify against

	// Tracing context
	TraceContext context.Context // OpenTelemetry trace context for span propagation
	UpstreamSpan trace.Span      // Span for tracking upstream vLLM request duration

	// Response API context
	ResponseAPICtx *ResponseAPIContext // Non-nil if this is a Response API request

	// Router replay context
	RouterReplayID       string                     // ID of the router replay session, if applicable
	RouterReplayConfig   *config.RouterReplayConfig // Configuration for router replay, if applicable
	RouterReplayRecorder *routerreplay.Recorder     // The recorder instance for this decision

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
}

// handleRequestHeaders processes the request headers
func (r *OpenAIRouter) handleRequestHeaders(v *ext_proc.ProcessingRequest_RequestHeaders, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	// Record start time for overall request processing
	ctx.StartTime = time.Now()

	// Initialize trace context from incoming headers
	baseCtx := context.Background()
	headerMap := make(map[string]string)
	for _, h := range v.RequestHeaders.Headers.Headers {
		headerValue := h.Value
		if headerValue == "" && len(h.RawValue) > 0 {
			headerValue = string(h.RawValue)
		}
		headerMap[h.Key] = headerValue
	}

	// Extract trace context from headers (if present)
	ctx.TraceContext = tracing.ExtractTraceContext(baseCtx, headerMap)

	// Start root span for the request
	spanCtx, span := tracing.StartSpan(ctx.TraceContext, tracing.SpanRequestReceived,
		trace.WithSpanKind(trace.SpanKindServer))
	ctx.TraceContext = spanCtx
	defer span.End()

	// Store headers for later use
	requestHeaders := v.RequestHeaders.Headers
	logging.Debugf("Processing request headers: %+v", requestHeaders.Headers)
	for _, h := range requestHeaders.Headers {
		// Prefer Value when available; fall back to RawValue
		headerValue := h.Value
		if headerValue == "" && len(h.RawValue) > 0 {
			headerValue = string(h.RawValue)
		}

		ctx.Headers[h.Key] = headerValue
		// Store request ID if present (case-insensitive)
		if strings.ToLower(h.Key) == headers.RequestID {
			ctx.RequestID = headerValue
		}
		// Check for looper request header
		if h.Key == headers.VSRLooperRequest && headerValue == "true" {
			ctx.LooperRequest = true
			logging.Infof("Detected looper internal request, will skip plugin processing")
		}
	}

	// Set request metadata on span
	if ctx.RequestID != "" {
		tracing.SetSpanAttributes(span,
			attribute.String(tracing.AttrRequestID, ctx.RequestID))
	}

	method := ctx.Headers[":method"]
	path := ctx.Headers[":path"]
	tracing.SetSpanAttributes(span,
		attribute.String(tracing.AttrHTTPMethod, method),
		attribute.String(tracing.AttrHTTPPath, path))
	// Router replay API (read-only): list or fetch replay records
	if replayResp := r.handleRouterReplayAPI(method, path); replayResp != nil {
		return replayResp, nil
	}

	// Detect if the client expects a streaming response (SSE)
	if accept, ok := ctx.Headers["accept"]; ok {
		if strings.Contains(strings.ToLower(accept), "text/event-stream") {
			ctx.ExpectStreamingResponse = true
			logging.Infof("Client expects streaming response based on Accept header")
		}
	}

	// Check if this is a GET request to /v1/models
	if method == "GET" && strings.HasPrefix(path, "/v1/models") {
		logging.Infof("Handling /v1/models request with path: %s", path)
		return r.handleModelsRequest(path)
	}

	// Handle Response API endpoints
	if r.ResponseAPIFilter != nil && r.ResponseAPIFilter.IsEnabled() && strings.HasPrefix(path, "/v1/responses") {
		// GET /v1/responses/{id}/input_items - Get input items for a response
		if method == "GET" && strings.HasSuffix(path, "/input_items") {
			responseID := extractResponseIDFromInputItemsPath(path)
			if responseID != "" {
				logging.Infof("Handling GET /v1/responses/%s/input_items", responseID)
				return r.ResponseAPIFilter.HandleGetInputItems(ctx.TraceContext, responseID)
			}
		}

		// GET /v1/responses/{id} - Get a response
		if method == "GET" {
			responseID := extractResponseIDFromPath(path)
			if responseID != "" {
				logging.Infof("Handling GET /v1/responses/%s", responseID)
				return r.ResponseAPIFilter.HandleGetResponse(ctx.TraceContext, responseID)
			}
		}

		// DELETE /v1/responses/{id} - Delete a response
		if method == "DELETE" {
			responseID := extractResponseIDFromPath(path)
			if responseID != "" {
				logging.Infof("Handling DELETE /v1/responses/%s", responseID)
				return r.ResponseAPIFilter.HandleDeleteResponse(ctx.TraceContext, responseID)
			}
		}

		// POST /v1/responses - Create response (mark for body phase processing)
		if method == "POST" {
			ctx.ResponseAPICtx = &ResponseAPIContext{IsResponseAPIRequest: true}
			logging.Infof("Detected Response API POST request: %s", path)
		}
	}

	// Prepare base response
	response := &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestHeaders{
			RequestHeaders: &ext_proc.HeadersResponse{
				Response: &ext_proc.CommonResponse{
					Status: ext_proc.CommonResponse_CONTINUE,
					// No HeaderMutation - will be handled in body phase
				},
			},
		},
	}

	// If streaming is expected, we rely on Envoy config to set response_body_mode: STREAMED for SSE.
	// Some Envoy/control-plane versions may not support per-message ModeOverride; avoid compile-time coupling here.
	// The Accept header is still recorded on context for downstream logic.

	return response, nil
}

// extractResponseIDFromPath extracts the response ID from a path like /v1/responses/{id}
func extractResponseIDFromPath(path string) string {
	// Remove query string if present
	if idx := strings.Index(path, "?"); idx != -1 {
		path = path[:idx]
	}

	// Expected format: /v1/responses/{id}
	prefix := "/v1/responses/"
	if !strings.HasPrefix(path, prefix) {
		return ""
	}

	id := strings.TrimPrefix(path, prefix)
	// Remove any trailing slashes
	id = strings.TrimSuffix(id, "/")

	// Skip if this is an input_items request
	if strings.Contains(id, "/") {
		return ""
	}

	// Validate it looks like a response ID (should start with "resp_")
	if id != "" && strings.HasPrefix(id, "resp_") {
		return id
	}

	return ""
}

// extractResponseIDFromInputItemsPath extracts the response ID from a path like /v1/responses/{id}/input_items
func extractResponseIDFromInputItemsPath(path string) string {
	// Remove query string if present
	if idx := strings.Index(path, "?"); idx != -1 {
		path = path[:idx]
	}

	// Expected format: /v1/responses/{id}/input_items
	prefix := "/v1/responses/"
	suffix := "/input_items"

	if !strings.HasPrefix(path, prefix) || !strings.HasSuffix(path, suffix) {
		return ""
	}

	// Extract the ID between prefix and suffix
	id := strings.TrimPrefix(path, prefix)
	id = strings.TrimSuffix(id, suffix)

	// Validate it looks like a response ID (should start with "resp_")
	if id != "" && strings.HasPrefix(id, "resp_") {
		return id
	}

	return ""
}
