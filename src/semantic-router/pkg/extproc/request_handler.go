package extproc

import (
	"context"
	"encoding/json"
	"strings"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/openai/openai-go"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/http"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/pii"
)

// parseOpenAIRequest parses the raw JSON using the OpenAI SDK types
func parseOpenAIRequest(data []byte) (*openai.ChatCompletionNewParams, error) {
	var req openai.ChatCompletionNewParams
	if err := json.Unmarshal(data, &req); err != nil {
		return nil, err
	}
	return &req, nil
}

// extractStreamParam extracts the stream parameter from the original request body
func extractStreamParam(originalBody []byte) bool {
	var requestMap map[string]interface{}
	if err := json.Unmarshal(originalBody, &requestMap); err != nil {
		return false
	}

	if streamValue, exists := requestMap["stream"]; exists {
		if stream, ok := streamValue.(bool); ok {
			return stream
		}
	}
	return false
}

// serializeOpenAIRequestWithStream converts request back to JSON, preserving the stream parameter from original request
func serializeOpenAIRequestWithStream(req *openai.ChatCompletionNewParams, hasStreamParam bool) ([]byte, error) {
	// First serialize the SDK object
	sdkBytes, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	// If original request had stream parameter, add it back
	if hasStreamParam {
		var sdkMap map[string]interface{}
		if err := json.Unmarshal(sdkBytes, &sdkMap); err == nil {
			sdkMap["stream"] = true
			if modifiedBytes, err := json.Marshal(sdkMap); err == nil {
				return modifiedBytes, nil
			}
		}
	}

	return sdkBytes, nil
}

// shouldClearRouteCache checks if route cache should be cleared
func (r *OpenAIRouter) shouldClearRouteCache() bool {
	// Check if feature is enabled
	return r.Config.ClearRouteCache
}

// addSystemPromptToRequestBody adds a system prompt to the beginning of the messages array in the JSON request body
// Returns the modified body, whether the system prompt was actually injected, and any error
func addSystemPromptToRequestBody(requestBody []byte, systemPrompt string, mode string) ([]byte, bool, error) {
	if systemPrompt == "" {
		return requestBody, false, nil
	}

	// Parse the JSON request body
	var requestMap map[string]interface{}
	if err := json.Unmarshal(requestBody, &requestMap); err != nil {
		return nil, false, err
	}

	// Get the messages array
	messagesInterface, ok := requestMap["messages"]
	if !ok {
		return requestBody, false, nil // No messages array, return original
	}

	messages, ok := messagesInterface.([]interface{})
	if !ok {
		return requestBody, false, nil // Messages is not an array, return original
	}

	// Check if there's already a system message at the beginning
	hasSystemMessage := false
	var existingSystemContent string
	if len(messages) > 0 {
		if firstMsg, ok := messages[0].(map[string]interface{}); ok {
			if role, ok := firstMsg["role"].(string); ok && role == "system" {
				hasSystemMessage = true
				if content, ok := firstMsg["content"].(string); ok {
					existingSystemContent = content
				}
			}
		}
	}

	// Handle different injection modes
	var finalSystemContent string
	var logMessage string

	switch mode {
	case "insert":
		if hasSystemMessage {
			// Insert mode: prepend category prompt to existing system message
			finalSystemContent = systemPrompt + "\n\n" + existingSystemContent
			logMessage = "Inserted category-specific system prompt before existing system message"
		} else {
			// No existing system message, just use the category prompt
			finalSystemContent = systemPrompt
			logMessage = "Added category-specific system prompt (insert mode, no existing system message)"
		}
	case "replace":
		fallthrough
	default:
		// Replace mode: use only the category prompt
		finalSystemContent = systemPrompt
		if hasSystemMessage {
			logMessage = "Replaced existing system message with category-specific system prompt"
		} else {
			logMessage = "Added category-specific system prompt to the beginning of messages"
		}
	}

	// Create the final system message
	systemMessage := map[string]interface{}{
		"role":    "system",
		"content": finalSystemContent,
	}

	if hasSystemMessage {
		// Update the existing system message
		messages[0] = systemMessage
	} else {
		// Prepend the system message to the beginning of the messages array
		messages = append([]interface{}{systemMessage}, messages...)
	}

	observability.Infof("%s (mode: %s)", logMessage, mode)

	// Update the messages in the request map
	requestMap["messages"] = messages

	// Marshal back to JSON
	modifiedBody, err := json.Marshal(requestMap)
	return modifiedBody, true, err
}

// extractUserAndNonUserContent extracts content from request messages
func extractUserAndNonUserContent(req *openai.ChatCompletionNewParams) (string, []string) {
	var userContent string
	var nonUser []string

	for _, msg := range req.Messages {
		// Extract content based on message type
		var textContent string
		var role string

		if msg.OfUser != nil {
			role = "user"
			// Handle user message content
			if msg.OfUser.Content.OfString.Value != "" {
				textContent = msg.OfUser.Content.OfString.Value
			} else if len(msg.OfUser.Content.OfArrayOfContentParts) > 0 {
				// Extract text from content parts
				var parts []string
				for _, part := range msg.OfUser.Content.OfArrayOfContentParts {
					if part.OfText != nil {
						parts = append(parts, part.OfText.Text)
					}
				}
				textContent = strings.Join(parts, " ")
			}
		} else if msg.OfSystem != nil {
			role = "system"
			if msg.OfSystem.Content.OfString.Value != "" {
				textContent = msg.OfSystem.Content.OfString.Value
			} else if len(msg.OfSystem.Content.OfArrayOfContentParts) > 0 {
				// Extract text from content parts
				var parts []string
				for _, part := range msg.OfSystem.Content.OfArrayOfContentParts {
					if part.Text != "" {
						parts = append(parts, part.Text)
					}
				}
				textContent = strings.Join(parts, " ")
			}
		} else if msg.OfAssistant != nil {
			role = "assistant"
			if msg.OfAssistant.Content.OfString.Value != "" {
				textContent = msg.OfAssistant.Content.OfString.Value
			} else if len(msg.OfAssistant.Content.OfArrayOfContentParts) > 0 {
				// Extract text from content parts
				var parts []string
				for _, part := range msg.OfAssistant.Content.OfArrayOfContentParts {
					if part.OfText != nil {
						parts = append(parts, part.OfText.Text)
					}
				}
				textContent = strings.Join(parts, " ")
			}
		}

		// Categorize by role
		if role == "user" {
			userContent = textContent
		} else if role != "" {
			nonUser = append(nonUser, textContent)
		}
	}

	return userContent, nonUser
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

	// TTFT tracking
	TTFTRecorded bool
	TTFTSeconds  float64

	// VSR decision tracking
	VSRSelectedCategory     string // The category selected by VSR
	VSRReasoningMode        string // "on" or "off" - whether reasoning mode was determined to be used
	VSRSelectedModel        string // The model selected by VSR
	VSRCacheHit             bool   // Whether this request hit the cache
	VSRInjectedSystemPrompt bool   // Whether a system prompt was injected into the request

	// Tracing context
	TraceContext context.Context // OpenTelemetry trace context for span propagation
}

// handleRequestHeaders processes the request headers
func (r *OpenAIRouter) handleRequestHeaders(v *ext_proc.ProcessingRequest_RequestHeaders, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	// Record start time for overall request processing
	ctx.StartTime = time.Now()
	observability.Infof("Received request headers")

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
	ctx.TraceContext = observability.ExtractTraceContext(baseCtx, headerMap)

	// Start root span for the request
	spanCtx, span := observability.StartSpan(ctx.TraceContext, observability.SpanRequestReceived,
		trace.WithSpanKind(trace.SpanKindServer))
	ctx.TraceContext = spanCtx
	defer span.End()

	// Store headers for later use
	requestHeaders := v.RequestHeaders.Headers
	for _, h := range requestHeaders.Headers {
		// Prefer Value when available; fall back to RawValue
		headerValue := h.Value
		if headerValue == "" && len(h.RawValue) > 0 {
			headerValue = string(h.RawValue)
		}
		observability.Debugf("Processing header: %s=%s", h.Key, headerValue)

		ctx.Headers[h.Key] = headerValue
		// Store request ID if present (case-insensitive)
		if strings.ToLower(h.Key) == headers.RequestID {
			ctx.RequestID = headerValue
		}
	}

	// Set request metadata on span
	if ctx.RequestID != "" {
		observability.SetSpanAttributes(span,
			attribute.String(observability.AttrRequestID, ctx.RequestID))
	}

	method := ctx.Headers[":method"]
	path := ctx.Headers[":path"]
	observability.SetSpanAttributes(span,
		attribute.String(observability.AttrHTTPMethod, method),
		attribute.String(observability.AttrHTTPPath, path))

	// Detect if the client expects a streaming response (SSE)
	if accept, ok := ctx.Headers["accept"]; ok {
		if strings.Contains(strings.ToLower(accept), "text/event-stream") {
			ctx.ExpectStreamingResponse = true
			observability.Infof("Client expects streaming response based on Accept header")
		}
	}

	// Check if this is a GET request to /v1/models
	if method == "GET" && strings.HasPrefix(path, "/v1/models") {
		observability.Infof("Handling /v1/models request with path: %s", path)
		return r.handleModelsRequest(path)
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

// handleRequestBody processes the request body
func (r *OpenAIRouter) handleRequestBody(v *ext_proc.ProcessingRequest_RequestBody, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	observability.Infof("Received request body %s", string(v.RequestBody.GetBody()))
	// Record start time for model routing
	ctx.ProcessingStartTime = time.Now()
	// Save the original request body
	ctx.OriginalRequestBody = v.RequestBody.GetBody()

	// Extract stream parameter from original request and update ExpectStreamingResponse if needed
	hasStreamParam := extractStreamParam(ctx.OriginalRequestBody)
	if hasStreamParam {
		observability.Infof("Original request contains stream parameter: true")
		ctx.ExpectStreamingResponse = true // Set this if stream param is found
	}

	// Parse the OpenAI request using SDK types
	openAIRequest, err := parseOpenAIRequest(ctx.OriginalRequestBody)
	if err != nil {
		observability.Errorf("Error parsing OpenAI request: %v", err)
		// Attempt to determine model for labeling (may be unknown here)
		metrics.RecordRequestError(ctx.RequestModel, "parse_error")
		// Count this request as well, with unknown model if necessary
		metrics.RecordModelRequest(ctx.RequestModel)
		return nil, status.Errorf(codes.InvalidArgument, "invalid request body: %v", err)
	}

	// Store the original model
	originalModel := openAIRequest.Model
	observability.Infof("Original model: %s", originalModel)

	// Set model on span
	if ctx.TraceContext != nil {
		_, span := observability.StartSpan(ctx.TraceContext, "parse_request")
		observability.SetSpanAttributes(span,
			attribute.String(observability.AttrOriginalModel, originalModel))
		span.End()
	}

	// Record the initial request to this model (count all requests)
	metrics.RecordModelRequest(originalModel)
	// Also set the model on context early so error metrics can label it
	if ctx.RequestModel == "" {
		ctx.RequestModel = originalModel
	}

	// Get content from messages
	userContent, nonUserMessages := extractUserAndNonUserContent(openAIRequest)

	// Perform security checks
	if response, shouldReturn := r.performSecurityChecks(ctx, userContent, nonUserMessages); shouldReturn {
		return response, nil
	}

	// Handle caching
	if response, shouldReturn := r.handleCaching(ctx); shouldReturn {
		return response, nil
	}

	// Handle model selection and routing
	return r.handleModelRouting(openAIRequest, originalModel, userContent, nonUserMessages, ctx)
}

// performSecurityChecks performs PII and jailbreak detection
func (r *OpenAIRouter) performSecurityChecks(ctx *RequestContext, userContent string, nonUserMessages []string) (*ext_proc.ProcessingResponse, bool) {
	// Perform PII classification on all message content
	allContent := pii.ExtractAllContent(userContent, nonUserMessages)

	// Perform jailbreak detection on all message content
	if r.Classifier.IsJailbreakEnabled() {
		// Start jailbreak detection span
		spanCtx, span := observability.StartSpan(ctx.TraceContext, observability.SpanJailbreakDetection)
		defer span.End()

		startTime := time.Now()
		hasJailbreak, jailbreakDetections, err := r.Classifier.AnalyzeContentForJailbreak(allContent)
		detectionTime := time.Since(startTime).Milliseconds()

		observability.SetSpanAttributes(span,
			attribute.Int64(observability.AttrJailbreakDetectionTimeMs, detectionTime))

		if err != nil {
			observability.Errorf("Error performing jailbreak analysis: %v", err)
			observability.RecordError(span, err)
			// Continue processing despite jailbreak analysis error
			metrics.RecordRequestError(ctx.RequestModel, "classification_failed")
		} else if hasJailbreak {
			// Find the first jailbreak detection for response
			var jailbreakType string
			var confidence float32
			for _, detection := range jailbreakDetections {
				if detection.IsJailbreak {
					jailbreakType = detection.JailbreakType
					confidence = detection.Confidence
					break
				}
			}

			observability.SetSpanAttributes(span,
				attribute.Bool(observability.AttrJailbreakDetected, true),
				attribute.String(observability.AttrJailbreakType, jailbreakType),
				attribute.String(observability.AttrSecurityAction, "blocked"))

			observability.Warnf("JAILBREAK ATTEMPT BLOCKED: %s (confidence: %.3f)", jailbreakType, confidence)

			// Return immediate jailbreak violation response
			// Structured log for security block
			observability.LogEvent("security_block", map[string]interface{}{
				"reason_code":    "jailbreak_detected",
				"jailbreak_type": jailbreakType,
				"confidence":     confidence,
				"request_id":     ctx.RequestID,
			})
			// Count this as a blocked request
			metrics.RecordRequestError(ctx.RequestModel, "jailbreak_block")
			jailbreakResponse := http.CreateJailbreakViolationResponse(jailbreakType, confidence, ctx.ExpectStreamingResponse)
			ctx.TraceContext = spanCtx
			return jailbreakResponse, true
		} else {
			observability.SetSpanAttributes(span,
				attribute.Bool(observability.AttrJailbreakDetected, false))
			observability.Infof("No jailbreak detected in request content")
			ctx.TraceContext = spanCtx
		}
	}

	return nil, false
}

// handleCaching handles cache lookup and storage
func (r *OpenAIRouter) handleCaching(ctx *RequestContext) (*ext_proc.ProcessingResponse, bool) {
	// Extract the model and query for cache lookup
	requestModel, requestQuery, err := cache.ExtractQueryFromOpenAIRequest(ctx.OriginalRequestBody)
	if err != nil {
		observability.Errorf("Error extracting query from request: %v", err)
		// Continue without caching
		return nil, false
	}

	ctx.RequestModel = requestModel
	ctx.RequestQuery = requestQuery

	if requestQuery != "" && r.Cache.IsEnabled() {
		// Start cache lookup span
		spanCtx, span := observability.StartSpan(ctx.TraceContext, observability.SpanCacheLookup)
		defer span.End()

		startTime := time.Now()
		// Try to find a similar cached response
		cachedResponse, found, cacheErr := r.Cache.FindSimilar(requestModel, requestQuery)
		lookupTime := time.Since(startTime).Milliseconds()

		observability.SetSpanAttributes(span,
			attribute.String(observability.AttrCacheKey, requestQuery),
			attribute.Bool(observability.AttrCacheHit, found),
			attribute.Int64(observability.AttrCacheLookupTimeMs, lookupTime))

		if cacheErr != nil {
			observability.Errorf("Error searching cache: %v", cacheErr)
			observability.RecordError(span, cacheErr)
		} else if found {
			// Mark this request as a cache hit
			ctx.VSRCacheHit = true
			// Log cache hit
			observability.LogEvent("cache_hit", map[string]interface{}{
				"request_id": ctx.RequestID,
				"model":      requestModel,
				"query":      requestQuery,
			})
			// Return immediate response from cache
			response := http.CreateCacheHitResponse(cachedResponse, ctx.ExpectStreamingResponse)
			ctx.TraceContext = spanCtx
			return response, true
		}
		ctx.TraceContext = spanCtx
	}

	// Cache miss, store the request for later
	err = r.Cache.AddPendingRequest(ctx.RequestID, requestModel, requestQuery, ctx.OriginalRequestBody)
	if err != nil {
		observability.Errorf("Error adding pending request to cache: %v", err)
		// Continue without caching
	}

	return nil, false
}

// handleModelRouting handles model selection and routing logic
func (r *OpenAIRouter) handleModelRouting(openAIRequest *openai.ChatCompletionNewParams, originalModel, userContent string, nonUserMessages []string, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	// Create default response with CONTINUE status
	response := &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status: ext_proc.CommonResponse_CONTINUE,
				},
			},
		},
	}

	// Only change the model if the original model is an auto model name (supports both "auto" and configured AutoModelName for backward compatibility)
	actualModel := originalModel
	var selectedEndpoint string
	isAutoModel := r.Config != nil && r.Config.IsAutoModelName(originalModel)
	if isAutoModel && (len(nonUserMessages) > 0 || userContent != "") {
		observability.Infof("Using Auto Model Selection (model=%s)", originalModel)
		// Determine text to use for classification/similarity
		var classificationText string
		if len(userContent) > 0 {
			classificationText = userContent
		} else if len(nonUserMessages) > 0 {
			// Fall back to user content if no system/assistant messages
			classificationText = strings.Join(nonUserMessages, " ")
		}

		if classificationText != "" {
			// Start classification span
			classifyCtx, classifySpan := observability.StartSpan(ctx.TraceContext, observability.SpanClassification)
			classifyStart := time.Now()

			// Find the most similar task description or classify, then select best model
			matchedModel := r.classifyAndSelectBestModel(classificationText)
			classifyTime := time.Since(classifyStart).Milliseconds()

			// Get category information for the span
			categoryName := r.findCategoryForClassification(classificationText)

			observability.SetSpanAttributes(classifySpan,
				attribute.String(observability.AttrCategoryName, categoryName),
				attribute.String(observability.AttrClassifierType, "bert"),
				attribute.Int64(observability.AttrClassificationTimeMs, classifyTime))
			classifySpan.End()
			ctx.TraceContext = classifyCtx

			if matchedModel != originalModel && matchedModel != "" {
				// Start PII detection span if enabled
				allContent := pii.ExtractAllContent(userContent, nonUserMessages)
				if r.PIIChecker.IsPIIEnabled(matchedModel) {
					piiCtx, piiSpan := observability.StartSpan(ctx.TraceContext, observability.SpanPIIDetection)
					piiStart := time.Now()

					observability.Infof("PII policy enabled for model %s", matchedModel)
					detectedPII := r.Classifier.DetectPIIInContent(allContent)

					piiTime := time.Since(piiStart).Milliseconds()
					piiDetected := len(detectedPII) > 0

					observability.SetSpanAttributes(piiSpan,
						attribute.Bool(observability.AttrPIIDetected, piiDetected),
						attribute.Int64(observability.AttrPIIDetectionTimeMs, piiTime))

					if piiDetected {
						// Convert detected PII to comma-separated string
						piiTypesStr := strings.Join(detectedPII, ",")
						observability.SetSpanAttributes(piiSpan,
							attribute.String(observability.AttrPIITypes, piiTypesStr))
					}

					piiSpan.End()
					ctx.TraceContext = piiCtx

					// Check if the initially selected model passes PII policy
					allowed, deniedPII, err := r.PIIChecker.CheckPolicy(matchedModel, detectedPII)
					if err != nil {
						observability.Errorf("Error checking PII policy for model %s: %v", matchedModel, err)
						// Continue with original selection on error
					} else if !allowed {
						observability.Warnf("Initially selected model %s violates PII policy, finding alternative", matchedModel)
						// Find alternative models from the same category that pass PII policy
						categoryName := r.findCategoryForClassification(classificationText)
						if categoryName != "" {
							alternativeModels := r.Classifier.GetModelsForCategory(categoryName)
							allowedModels := r.PIIChecker.FilterModelsForPII(alternativeModels, detectedPII)
							if len(allowedModels) > 0 {
								// Select the best allowed model from this category
								matchedModel = r.Classifier.SelectBestModelFromList(allowedModels, categoryName)
								observability.Infof("Selected alternative model %s that passes PII policy", matchedModel)
								// Record reason code for selecting alternative due to PII
								metrics.RecordRoutingReasonCode("pii_policy_alternative_selected", matchedModel)
							} else {
								observability.Warnf("No models in category %s pass PII policy, using default", categoryName)
								matchedModel = r.Config.DefaultModel
								// Check if default model passes policy
								defaultAllowed, defaultDeniedPII, _ := r.PIIChecker.CheckPolicy(matchedModel, detectedPII)
								if !defaultAllowed {
									observability.Errorf("Default model also violates PII policy, returning error")
									observability.LogEvent("routing_block", map[string]interface{}{
										"reason_code": "pii_policy_denied_default_model",
										"request_id":  ctx.RequestID,
										"model":       matchedModel,
										"denied_pii":  defaultDeniedPII,
									})
									metrics.RecordRequestError(matchedModel, "pii_policy_denied")
									piiResponse := http.CreatePIIViolationResponse(matchedModel, defaultDeniedPII, ctx.ExpectStreamingResponse)
									return piiResponse, nil
								}
							}
						} else {
							observability.Warnf("Could not determine category, returning PII violation for model %s", matchedModel)
							observability.LogEvent("routing_block", map[string]interface{}{
								"reason_code": "pii_policy_denied",
								"request_id":  ctx.RequestID,
								"model":       matchedModel,
								"denied_pii":  deniedPII,
							})
							metrics.RecordRequestError(matchedModel, "pii_policy_denied")
							piiResponse := http.CreatePIIViolationResponse(matchedModel, deniedPII, ctx.ExpectStreamingResponse)
							return piiResponse, nil
						}
					}
				}

				observability.Infof("Routing to model: %s", matchedModel)

				// Start routing decision span
				routingCtx, routingSpan := observability.StartSpan(ctx.TraceContext, observability.SpanRoutingDecision)

				// Check reasoning mode for this category using entropy-based approach
				useReasoning, categoryName, reasoningDecision := r.getEntropyBasedReasoningModeAndCategory(userContent)
				observability.Infof("Entropy-based reasoning decision for this query: %v on [%s] model (confidence: %.3f, reason: %s)",
					useReasoning, matchedModel, reasoningDecision.Confidence, reasoningDecision.DecisionReason)
				// Record reasoning decision metric with the effort that will be applied if enabled
				effortForMetrics := r.getReasoningEffort(categoryName, matchedModel)
				metrics.RecordReasoningDecision(categoryName, matchedModel, useReasoning, effortForMetrics)

				// Set routing attributes on span
				observability.SetSpanAttributes(routingSpan,
					attribute.String(observability.AttrRoutingStrategy, "auto"),
					attribute.String(observability.AttrRoutingReason, reasoningDecision.DecisionReason),
					attribute.String(observability.AttrOriginalModel, originalModel),
					attribute.String(observability.AttrSelectedModel, matchedModel),
					attribute.Bool(observability.AttrReasoningEnabled, useReasoning),
					attribute.String(observability.AttrReasoningEffort, effortForMetrics))

				routingSpan.End()
				ctx.TraceContext = routingCtx

				// Track VSR decision information
				ctx.VSRSelectedCategory = categoryName
				ctx.VSRSelectedModel = matchedModel
				if useReasoning {
					ctx.VSRReasoningMode = "on"
				} else {
					ctx.VSRReasoningMode = "off"
				}

				// Track the model routing change
				metrics.RecordModelRouting(originalModel, matchedModel)

				// Update the actual model that will be used
				actualModel = matchedModel

				// Start backend selection span
				backendCtx, backendSpan := observability.StartSpan(ctx.TraceContext, observability.SpanBackendSelection)

				// Select the best endpoint for this model
				endpointAddress, endpointFound := r.Config.SelectBestEndpointAddressForModel(matchedModel)
				if endpointFound {
					selectedEndpoint = endpointAddress
					observability.Infof("Selected endpoint address: %s for model: %s", selectedEndpoint, matchedModel)

					// Extract endpoint name from config
					endpoints := r.Config.GetEndpointsForModel(matchedModel)
					if len(endpoints) > 0 {
						observability.SetSpanAttributes(backendSpan,
							attribute.String(observability.AttrEndpointName, endpoints[0].Name),
							attribute.String(observability.AttrEndpointAddress, selectedEndpoint))
					}
				} else {
					observability.Warnf("No endpoint found for model %s, using fallback", matchedModel)
				}

				backendSpan.End()
				ctx.TraceContext = backendCtx

				// Modify the model in the request
				openAIRequest.Model = matchedModel

				// Serialize the modified request with stream parameter preserved
				modifiedBody, err := serializeOpenAIRequestWithStream(openAIRequest, ctx.ExpectStreamingResponse)
				if err != nil {
					observability.Errorf("Error serializing modified request: %v", err)
					metrics.RecordRequestError(actualModel, "serialization_error")
					return nil, status.Errorf(codes.Internal, "error serializing modified request: %v", err)
				}

				modifiedBody, err = r.setReasoningModeToRequestBody(modifiedBody, useReasoning, categoryName)
				if err != nil {
					observability.Errorf("Error setting reasoning mode %v to request: %v", useReasoning, err)
					metrics.RecordRequestError(actualModel, "serialization_error")
					return nil, status.Errorf(codes.Internal, "error setting reasoning mode: %v", err)
				}

				// Add category-specific system prompt if configured
				if categoryName != "" {
					// Try to get the most up-to-date category configuration from global config first
					// This ensures API updates are reflected immediately
					globalConfig := config.GetConfig()
					var category *config.Category
					if globalConfig != nil {
						category = globalConfig.GetCategoryByName(categoryName)
					}

					// If not found in global config, fall back to router's config (for tests and initial setup)
					if category == nil {
						category = r.Classifier.GetCategoryByName(categoryName)
					}

					if category != nil && category.SystemPrompt != "" && category.IsSystemPromptEnabled() {
						// Start system prompt injection span
						promptCtx, promptSpan := observability.StartSpan(ctx.TraceContext, observability.SpanSystemPromptInjection)

						mode := category.GetSystemPromptMode()
						var injected bool
						modifiedBody, injected, err = addSystemPromptToRequestBody(modifiedBody, category.SystemPrompt, mode)
						if err != nil {
							observability.Errorf("Error adding system prompt to request: %v", err)
							observability.RecordError(promptSpan, err)
							metrics.RecordRequestError(actualModel, "serialization_error")
							promptSpan.End()
							return nil, status.Errorf(codes.Internal, "error adding system prompt: %v", err)
						}

						observability.SetSpanAttributes(promptSpan,
							attribute.Bool("system_prompt.injected", injected),
							attribute.String("system_prompt.mode", mode),
							attribute.String(observability.AttrCategoryName, categoryName))

						if injected {
							ctx.VSRInjectedSystemPrompt = true
							observability.Infof("Added category-specific system prompt for category: %s (mode: %s)", categoryName, mode)
						}

						// Log metadata about system prompt injection (avoid logging sensitive user data)
						observability.Infof("System prompt injection completed for category: %s, body size: %d bytes", categoryName, len(modifiedBody))

						promptSpan.End()
						ctx.TraceContext = promptCtx
					} else if category != nil && category.SystemPrompt != "" && !category.IsSystemPromptEnabled() {
						observability.Infof("System prompt disabled for category: %s", categoryName)
					}
				}

				// Create body mutation with the modified body
				bodyMutation := &ext_proc.BodyMutation{
					Mutation: &ext_proc.BodyMutation_Body{
						Body: modifiedBody,
					},
				}

				// Create header mutation with content-length removal AND all necessary routing headers
				// (body phase HeaderMutation replaces header phase completely)
				setHeaders := []*core.HeaderValueOption{}
				if selectedEndpoint != "" {
					setHeaders = append(setHeaders, &core.HeaderValueOption{
						Header: &core.HeaderValue{
							Key:      headers.GatewayDestinationEndpoint,
							RawValue: []byte(selectedEndpoint),
						},
					})
				}
				if actualModel != "" {
					setHeaders = append(setHeaders, &core.HeaderValueOption{
						Header: &core.HeaderValue{
							Key:      headers.SelectedModel,
							RawValue: []byte(actualModel),
						},
					})
				}

				headerMutation := &ext_proc.HeaderMutation{
					RemoveHeaders: []string{"content-length"},
					SetHeaders:    setHeaders,
				}

				observability.Debugf("ActualModel = '%s'", actualModel)

				// Set the response with body mutation and content-length removal
				response = &ext_proc.ProcessingResponse{
					Response: &ext_proc.ProcessingResponse_RequestBody{
						RequestBody: &ext_proc.BodyResponse{
							Response: &ext_proc.CommonResponse{
								Status:         ext_proc.CommonResponse_CONTINUE,
								HeaderMutation: headerMutation,
								BodyMutation:   bodyMutation,
							},
						},
					},
				}

				observability.Infof("Use new model: %s", matchedModel)

				// Structured log for routing decision (auto)
				observability.LogEvent("routing_decision", map[string]interface{}{
					"reason_code":        "auto_routing",
					"request_id":         ctx.RequestID,
					"original_model":     originalModel,
					"selected_model":     matchedModel,
					"category":           categoryName,
					"reasoning_enabled":  useReasoning,
					"reasoning_effort":   effortForMetrics,
					"selected_endpoint":  selectedEndpoint,
					"routing_latency_ms": time.Since(ctx.ProcessingStartTime).Milliseconds(),
				})
				metrics.RecordRoutingReasonCode("auto_routing", matchedModel)
			}
		}
	} else if !isAutoModel {
		observability.Infof("Using specified model: %s", originalModel)
		// Track VSR decision information for non-auto models
		ctx.VSRSelectedModel = originalModel
		ctx.VSRReasoningMode = "off" // Non-auto models don't use reasoning mode by default
		// For non-auto models, check PII policy compliance
		allContent := pii.ExtractAllContent(userContent, nonUserMessages)
		detectedPII := r.Classifier.DetectPIIInContent(allContent)

		allowed, deniedPII, err := r.PIIChecker.CheckPolicy(originalModel, detectedPII)
		if err != nil {
			observability.Errorf("Error checking PII policy for model %s: %v", originalModel, err)
			// Continue with request on error
		} else if !allowed {
			observability.Errorf("Model %s violates PII policy, returning error", originalModel)
			observability.LogEvent("routing_block", map[string]interface{}{
				"reason_code": "pii_policy_denied",
				"request_id":  ctx.RequestID,
				"model":       originalModel,
				"denied_pii":  deniedPII,
			})
			metrics.RecordRequestError(originalModel, "pii_policy_denied")
			piiResponse := http.CreatePIIViolationResponse(originalModel, deniedPII, ctx.ExpectStreamingResponse)
			return piiResponse, nil
		}

		// Select the best endpoint for the specified model
		endpointAddress, endpointFound := r.Config.SelectBestEndpointAddressForModel(originalModel)
		if endpointFound {
			selectedEndpoint = endpointAddress
			observability.Infof("Selected endpoint address: %s for model: %s", selectedEndpoint, originalModel)
		} else {
			observability.Warnf("No endpoint found for model %s, using fallback", originalModel)
		}
		setHeaders := []*core.HeaderValueOption{}
		if selectedEndpoint != "" {
			setHeaders = append(setHeaders, &core.HeaderValueOption{
				Header: &core.HeaderValue{
					Key:      headers.GatewayDestinationEndpoint,
					RawValue: []byte(selectedEndpoint),
				},
			})
		}
		// Set x-selected-model header for non-auto models
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      "x-selected-model",
				RawValue: []byte(originalModel),
			},
		})
		// Create CommonResponse with cache clearing if enabled
		commonResponse := &ext_proc.CommonResponse{
			Status: ext_proc.CommonResponse_CONTINUE,
			HeaderMutation: &ext_proc.HeaderMutation{
				SetHeaders: setHeaders,
			},
		}

		// Check if route cache should be cleared
		if r.shouldClearRouteCache() {
			commonResponse.ClearRouteCache = true
		}

		// Set the response with body mutation and content-length removal
		response = &ext_proc.ProcessingResponse{
			Response: &ext_proc.ProcessingResponse_RequestBody{
				RequestBody: &ext_proc.BodyResponse{
					Response: commonResponse,
				},
			},
		}
		// Structured log for routing decision (explicit model)
		observability.LogEvent("routing_decision", map[string]interface{}{
			"reason_code":        "model_specified",
			"request_id":         ctx.RequestID,
			"original_model":     originalModel,
			"selected_model":     originalModel,
			"category":           "",
			"reasoning_enabled":  false,
			"reasoning_effort":   "",
			"selected_endpoint":  selectedEndpoint,
			"routing_latency_ms": time.Since(ctx.ProcessingStartTime).Milliseconds(),
		})
		metrics.RecordRoutingReasonCode("model_specified", originalModel)
	}

	// Check if route cache should be cleared (only for auto models, non-auto models handle this in their own path)
	// isAutoModel already determined at the beginning of this function using IsAutoModelName
	if isAutoModel && r.shouldClearRouteCache() {
		// Access the CommonResponse that's already created in this function
		if response.GetRequestBody() != nil && response.GetRequestBody().GetResponse() != nil {
			response.GetRequestBody().GetResponse().ClearRouteCache = true
			observability.Debugf("Setting ClearRouteCache=true (feature enabled) for auto model")
		}
	}

	// Save the actual model that will be used for token tracking
	ctx.RequestModel = actualModel

	// Handle tool selection based on tool_choice field
	if err := r.handleToolSelection(openAIRequest, userContent, nonUserMessages, &response, ctx); err != nil {
		observability.Errorf("Error in tool selection: %v", err)
		// Continue without failing the request
	}

	// Record the routing latency
	routingLatency := time.Since(ctx.ProcessingStartTime)
	metrics.RecordModelRoutingLatency(routingLatency.Seconds())

	return response, nil
}

// handleToolSelection handles automatic tool selection based on semantic similarity
func (r *OpenAIRouter) handleToolSelection(openAIRequest *openai.ChatCompletionNewParams, userContent string, nonUserMessages []string, response **ext_proc.ProcessingResponse, ctx *RequestContext) error {
	// Check if tool_choice is set to "auto"
	if openAIRequest.ToolChoice.OfAuto.Value == "auto" {
		// Continue with tool selection logic
	} else {
		return nil // Not auto tool selection
	}

	// Get text for tools classification
	var classificationText string
	if len(userContent) > 0 {
		classificationText = userContent
	} else if len(nonUserMessages) > 0 {
		classificationText = strings.Join(nonUserMessages, " ")
	}

	if classificationText == "" {
		observability.Infof("No content available for tool classification")
		return nil
	}

	if !r.ToolsDatabase.IsEnabled() {
		observability.Infof("Tools database is disabled")
		return nil
	}

	// Get configuration for tool selection
	topK := r.Config.Tools.TopK
	if topK <= 0 {
		topK = 3 // Default to 3 tools
	}

	// Find similar tools based on the query
	selectedTools, err := r.ToolsDatabase.FindSimilarTools(classificationText, topK)
	if err != nil {
		if r.Config.Tools.FallbackToEmpty {
			observability.Warnf("Tool selection failed, falling back to no tools: %v", err)
			openAIRequest.Tools = nil
			return r.updateRequestWithTools(openAIRequest, response, ctx)
		}
		metrics.RecordRequestError(getModelFromCtx(ctx), "classification_failed")
		return err
	}

	if len(selectedTools) == 0 {
		if r.Config.Tools.FallbackToEmpty {
			observability.Infof("No suitable tools found, falling back to no tools")
			openAIRequest.Tools = nil
		} else {
			observability.Infof("No suitable tools found above threshold")
			openAIRequest.Tools = []openai.ChatCompletionToolParam{} // Empty array
		}
	} else {
		// Convert selected tools to OpenAI SDK tool format
		tools := make([]openai.ChatCompletionToolParam, len(selectedTools))
		for i, tool := range selectedTools {
			// Convert the tool to OpenAI SDK format
			toolBytes, err := json.Marshal(tool)
			if err != nil {
				metrics.RecordRequestError(getModelFromCtx(ctx), "serialization_error")
				return err
			}
			var sdkTool openai.ChatCompletionToolParam
			if err := json.Unmarshal(toolBytes, &sdkTool); err != nil {
				return err
			}
			tools[i] = sdkTool
		}

		openAIRequest.Tools = tools
		observability.Infof("Auto-selected %d tools for query: %s", len(selectedTools), classificationText)
	}

	return r.updateRequestWithTools(openAIRequest, response, ctx)
}

// updateRequestWithTools updates the request body with the selected tools
func (r *OpenAIRouter) updateRequestWithTools(openAIRequest *openai.ChatCompletionNewParams, response **ext_proc.ProcessingResponse, ctx *RequestContext) error {
	// Re-serialize the request with modified tools and preserved stream parameter
	modifiedBody, err := serializeOpenAIRequestWithStream(openAIRequest, ctx.ExpectStreamingResponse)
	if err != nil {
		return err
	}

	// Create body mutation with the modified body
	bodyMutation := &ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_Body{
			Body: modifiedBody,
		},
	}

	// Create header mutation with content-length removal AND all necessary routing headers
	// (body phase HeaderMutation replaces header phase completely)

	// Get the headers that should have been set in the main routing
	var selectedEndpoint, actualModel string

	// These should be available from the existing response
	if (*response).GetRequestBody() != nil && (*response).GetRequestBody().GetResponse() != nil &&
		(*response).GetRequestBody().GetResponse().GetHeaderMutation() != nil &&
		(*response).GetRequestBody().GetResponse().GetHeaderMutation().GetSetHeaders() != nil {
		for _, header := range (*response).GetRequestBody().GetResponse().GetHeaderMutation().GetSetHeaders() {
			switch header.Header.Key {
			case headers.GatewayDestinationEndpoint:
				selectedEndpoint = header.Header.Value
			case headers.SelectedModel:
				actualModel = header.Header.Value
			}
		}
	}

	setHeaders := []*core.HeaderValueOption{}
	if selectedEndpoint != "" {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.GatewayDestinationEndpoint,
				RawValue: []byte(selectedEndpoint),
			},
		})
	}
	if actualModel != "" {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      headers.SelectedModel,
				RawValue: []byte(actualModel),
			},
		})
	}

	// Intentionally do not mutate Authorization header here

	headerMutation := &ext_proc.HeaderMutation{
		RemoveHeaders: []string{"content-length"},
		SetHeaders:    setHeaders,
	}

	// Create CommonResponse
	commonResponse := &ext_proc.CommonResponse{
		Status:         ext_proc.CommonResponse_CONTINUE,
		HeaderMutation: headerMutation,
		BodyMutation:   bodyMutation,
	}

	// Check if route cache should be cleared
	if r.shouldClearRouteCache() {
		commonResponse.ClearRouteCache = true
		observability.Debugf("Setting ClearRouteCache=true (feature enabled) in updateRequestWithTools")
	}

	// Update the response with body mutation and content-length removal
	*response = &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: commonResponse,
			},
		},
	}

	return nil
}

// OpenAIModel represents a single model in the OpenAI /v1/models response
type OpenAIModel struct {
	ID          string `json:"id"`
	Object      string `json:"object"`
	Created     int64  `json:"created"`
	OwnedBy     string `json:"owned_by"`
	Description string `json:"description,omitempty"` // Optional description for Chat UI
	LogoURL     string `json:"logo_url,omitempty"`    // Optional logo URL for Chat UI
}

// OpenAIModelList is the container for the models list response
type OpenAIModelList struct {
	Object string        `json:"object"`
	Data   []OpenAIModel `json:"data"`
}

// handleModelsRequest handles GET /v1/models requests and returns a direct response
// Whether to include configured models is controlled by the config's IncludeConfigModelsInList setting (default: false)
func (r *OpenAIRouter) handleModelsRequest(_ string) (*ext_proc.ProcessingResponse, error) {
	now := time.Now().Unix()

	// Start with the configured auto model name (or default "MoM")
	// The model list uses the actual configured name, not "auto"
	// However, "auto" is still accepted as an alias in request handling for backward compatibility
	models := []OpenAIModel{}

	// Add the effective auto model name (configured or default "MoM")
	if r.Config != nil {
		effectiveAutoModelName := r.Config.GetEffectiveAutoModelName()
		models = append(models, OpenAIModel{
			ID:          effectiveAutoModelName,
			Object:      "model",
			Created:     now,
			OwnedBy:     "vllm-semantic-router",
			Description: "Intelligent Router for Mixture-of-Models",
			LogoURL:     "https://github.com/vllm-project/semantic-router/blob/main/website/static/img/vllm.png", // You can customize this URL
		})
	} else {
		// Fallback if no config
		models = append(models, OpenAIModel{
			ID:          "MoM",
			Object:      "model",
			Created:     now,
			OwnedBy:     "vllm-semantic-router",
			Description: "Intelligent Router for Mixture-of-Models",
			LogoURL:     "https://github.com/vllm-project/semantic-router/blob/main/website/static/img/vllm.png", // You can customize this URL
		})
	}

	// Append underlying models from config (if available and configured to include them)
	if r.Config != nil && r.Config.IncludeConfigModelsInList {
		for _, m := range r.Config.GetAllModels() {
			// Skip if already added as the configured auto model name (avoid duplicates)
			if m == r.Config.GetEffectiveAutoModelName() {
				continue
			}
			models = append(models, OpenAIModel{
				ID:      m,
				Object:  "model",
				Created: now,
				OwnedBy: "vllm-semantic-router",
			})
		}
	}

	resp := OpenAIModelList{
		Object: "list",
		Data:   models,
	}

	return r.createJSONResponse(200, resp), nil
}

// statusCodeToEnum converts HTTP status code to typev3.StatusCode enum
func statusCodeToEnum(statusCode int) typev3.StatusCode {
	switch statusCode {
	case 200:
		return typev3.StatusCode_OK
	case 400:
		return typev3.StatusCode_BadRequest
	case 404:
		return typev3.StatusCode_NotFound
	case 500:
		return typev3.StatusCode_InternalServerError
	default:
		return typev3.StatusCode_OK
	}
}

// createJSONResponseWithBody creates a direct response with pre-marshaled JSON body
func (r *OpenAIRouter) createJSONResponseWithBody(statusCode int, jsonBody []byte) *ext_proc.ProcessingResponse {
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &ext_proc.ImmediateResponse{
				Status: &typev3.HttpStatus{
					Code: statusCodeToEnum(statusCode),
				},
				Headers: &ext_proc.HeaderMutation{
					SetHeaders: []*core.HeaderValueOption{
						{
							Header: &core.HeaderValue{
								Key:      "content-type",
								RawValue: []byte("application/json"),
							},
						},
					},
				},
				Body: jsonBody,
			},
		},
	}
}

// createJSONResponse creates a direct response with JSON content
func (r *OpenAIRouter) createJSONResponse(statusCode int, data interface{}) *ext_proc.ProcessingResponse {
	jsonData, err := json.Marshal(data)
	if err != nil {
		observability.Errorf("Failed to marshal JSON response: %v", err)
		return r.createErrorResponse(500, "Internal server error")
	}

	return r.createJSONResponseWithBody(statusCode, jsonData)
}

// createErrorResponse creates a direct error response
func (r *OpenAIRouter) createErrorResponse(statusCode int, message string) *ext_proc.ProcessingResponse {
	errorResp := map[string]interface{}{
		"error": map[string]interface{}{
			"message": message,
			"type":    "invalid_request_error",
			"code":    statusCode,
		},
	}

	jsonData, err := json.Marshal(errorResp)
	if err != nil {
		observability.Errorf("Failed to marshal error response: %v", err)
		jsonData = []byte(`{"error":{"message":"Internal server error","type":"internal_error","code":500}}`)
		// Use 500 status code for fallback error
		statusCode = 500
	}

	return r.createJSONResponseWithBody(statusCode, jsonData)
}
