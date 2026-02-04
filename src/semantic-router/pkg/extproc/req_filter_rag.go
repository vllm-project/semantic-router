package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
)

// executeRAGPlugin executes the RAG plugin if enabled for the decision
// Returns error if retrieval fails and on_failure is "block", nil otherwise
func (r *OpenAIRouter) executeRAGPlugin(ctx *RequestContext, decisionName string) error {
	// Skip RAG retrieval for looper internal requests
	// RAG context should only be retrieved once for the original request
	if ctx.LooperRequest {
		logging.Debugf("[RAG] Skipping RAG retrieval for looper internal request")
		return nil
	}

	// Get decision
	decision := ctx.VSRSelectedDecision
	if decision == nil {
		return nil // No decision, skip RAG
	}

	// Get RAG config
	ragConfig := decision.GetRAGConfig()
	if ragConfig == nil || !ragConfig.Enabled {
		return nil // RAG not enabled for this decision
	}

	// Check minimum confidence threshold
	if ragConfig.MinConfidenceThreshold != nil {
		confidence := ctx.FactCheckConfidence
		if confidence < *ragConfig.MinConfidenceThreshold {
			logging.Debugf("RAG skipped: confidence %.3f < threshold %.3f",
				confidence, *ragConfig.MinConfidenceThreshold)
			return nil
		}
	}

	// Start tracing
	ragCtx, ragSpan := tracing.StartSpan(ctx.TraceContext, tracing.SpanRAGRetrieval)
	defer ragSpan.End()

	// Set base attributes
	attributes := []attribute.KeyValue{
		attribute.String("rag.backend", ragConfig.Backend),
		attribute.String("rag.decision", decisionName),
	}

	// Add optional attributes if present
	if ragConfig.SimilarityThreshold != nil {
		attributes = append(attributes, attribute.Float64("rag.similarity_threshold", float64(*ragConfig.SimilarityThreshold)))
	}
	if ragConfig.TopK != nil {
		attributes = append(attributes, attribute.Int("rag.top_k", *ragConfig.TopK))
	}

	tracing.SetSpanAttributes(ragSpan, attributes...)

	// Perform retrieval
	start := time.Now()
	retrievedContext, err := r.retrieveContext(ragCtx, ctx, ragConfig)
	latency := time.Since(start).Seconds()
	ctx.RAGRetrievalLatency = latency

	tracing.SetSpanAttributes(ragSpan,
		attribute.Float64("rag.latency_seconds", latency),
		attribute.Bool("rag.success", err == nil),
	)

	if err != nil {
		tracing.RecordError(ragSpan, err)
		ragSpan.SetStatus(codes.Error, err.Error())
		metrics.RecordRAGRetrieval(ragConfig.Backend, decisionName, "error", latency)

		// Handle failure based on on_failure setting
		onFailure := ragConfig.OnFailure
		if onFailure == "" {
			onFailure = "skip" // Default
		}

		switch onFailure {
		case "block":
			logging.Errorf("RAG retrieval failed and on_failure=block: %v", err)
			return fmt.Errorf("RAG retrieval failed: %w", err)
		case "warn":
			logging.Warnf("RAG retrieval failed (on_failure=warn): %v", err)
			// Continue with warning - will be added to response headers
			ctx.RAGRetrievedContext = "" // Clear any partial context
		case "skip":
			fallthrough
		default:
			logging.Infof("RAG retrieval failed (on_failure=skip), continuing without context: %v", err)
			ctx.RAGRetrievedContext = "" // Clear any partial context
		}
		return nil
	}

	// Record success metrics
	tracing.SetSpanAttributes(ragSpan,
		attribute.Int("rag.context_length", len(retrievedContext)),
	)
	if ctx.RAGSimilarityScore > 0 {
		tracing.SetSpanAttributes(ragSpan, attribute.Float64("rag.similarity_score", float64(ctx.RAGSimilarityScore)))
	}

	ragSpan.SetStatus(codes.Ok, "Retrieval successful")
	metrics.RecordRAGRetrieval(ragConfig.Backend, decisionName, "success", latency)
	metrics.RecordRAGContextLength(ragConfig.Backend, decisionName, len(retrievedContext))
	if ctx.RAGSimilarityScore > 0 {
		metrics.RecordRAGSimilarityScore(ragConfig.Backend, decisionName, ctx.RAGSimilarityScore)
	}

	// Inject context into request
	if err := r.injectRAGContext(ctx, retrievedContext, ragConfig); err != nil {
		logging.Errorf("Failed to inject RAG context: %v", err)
		// Don't fail the request, just log the error
		return nil
	}

	logging.Infof("RAG plugin executed successfully: backend=%s, context_len=%d, latency=%.3fs",
		ragConfig.Backend, len(retrievedContext), latency)

	return nil
}

// retrieveContext retrieves context from the configured backend
func (r *OpenAIRouter) retrieveContext(traceCtx context.Context, ctx *RequestContext, ragConfig *config.RAGPluginConfig) (string, error) {
	// Check cache first if enabled
	if ragConfig.CacheResults {
		if cached, found := r.getRAGCache(ctx.UserContent, ragConfig); found {
			metrics.RecordRAGCacheHit(ragConfig.Backend)
			logging.Debugf("RAG cache hit for query: %s", ctx.UserContent[:min(50, len(ctx.UserContent))])
			return cached, nil
		}
		metrics.RecordRAGCacheMiss(ragConfig.Backend)
	}

	// Perform retrieval based on backend type
	var retrievedContext string
	var err error

	switch ragConfig.Backend {
	case "milvus":
		retrievedContext, err = r.retrieveFromMilvus(traceCtx, ctx, ragConfig)
	case "external_api":
		retrievedContext, err = r.retrieveFromExternalAPI(traceCtx, ctx, ragConfig)
	case "mcp":
		retrievedContext, err = r.retrieveFromMCP(traceCtx, ctx, ragConfig)
	case "openai":
		retrievedContext, err = r.retrieveFromOpenAI(traceCtx, ctx, ragConfig)
	case "hybrid":
		retrievedContext, err = r.retrieveFromHybrid(traceCtx, ctx, ragConfig)
	default:
		return "", fmt.Errorf("unknown RAG backend: %s", ragConfig.Backend)
	}

	if err != nil {
		return "", err
	}

	// Store backend info
	ctx.RAGBackend = ragConfig.Backend

	// Cache result if enabled
	if ragConfig.CacheResults && retrievedContext != "" {
		r.setRAGCache(ctx.UserContent, retrievedContext, ragConfig)
	}

	return retrievedContext, nil
}

// injectRAGContext injects retrieved context into the request
func (r *OpenAIRouter) injectRAGContext(ctx *RequestContext, retrievedContext string, ragConfig *config.RAGPluginConfig) error {
	if retrievedContext == "" {
		return nil
	}

	// Parse request body
	var requestMap map[string]interface{}
	if err := json.Unmarshal(ctx.OriginalRequestBody, &requestMap); err != nil {
		return fmt.Errorf("failed to parse request: %w", err)
	}

	// Get messages array
	messagesInterface, ok := requestMap["messages"]
	if !ok {
		return fmt.Errorf("messages array not found")
	}

	messages, ok := messagesInterface.([]interface{})
	if !ok {
		return fmt.Errorf("messages is not an array")
	}

	// Determine injection mode
	injectionMode := ragConfig.InjectionMode
	if injectionMode == "" {
		injectionMode = "tool_role" // Default
	}

	// Truncate context if needed (preserving UTF-8 character boundaries)
	maxLength := 10000 // Default
	if ragConfig.MaxContextLength != nil {
		maxLength = *ragConfig.MaxContextLength
	}
	if len([]rune(retrievedContext)) > maxLength {
		runes := []rune(retrievedContext)
		retrievedContext = string(runes[:maxLength]) + "..."
		logging.Infof("RAG context truncated to %d characters", maxLength)
	}

	switch injectionMode {
	case "tool_role":
		return r.injectAsToolRole(messages, retrievedContext, requestMap, ctx)
	case "system_prompt":
		return r.injectAsSystemPrompt(messages, retrievedContext, requestMap, ctx)
	default:
		return fmt.Errorf("unknown injection mode: %s", injectionMode)
	}
}

// injectAsToolRole injects context as tool role messages
func (r *OpenAIRouter) injectAsToolRole(messages []interface{}, context string, requestMap map[string]interface{}, ctx *RequestContext) error {
	// Find last user message
	lastUserIdx := -1
	for i := len(messages) - 1; i >= 0; i-- {
		msg, ok := messages[i].(map[string]interface{})
		if !ok {
			continue
		}
		role, _ := msg["role"].(string)
		if role == "user" {
			lastUserIdx = i
			break
		}
	}

	if lastUserIdx == -1 {
		return fmt.Errorf("no user message found")
	}

	// Create tool role message with unique ID
	// Use timestamp + request ID for better uniqueness
	toolCallID := fmt.Sprintf("rag_%d", time.Now().UnixNano())
	if ctx.RequestID != "" {
		toolCallID = fmt.Sprintf("rag_%s_%d", ctx.RequestID, time.Now().UnixNano())
	}
	toolMessage := map[string]interface{}{
		"role":         "tool",
		"tool_call_id": toolCallID,
		"content":      context,
	}

	// Insert after last user message
	newMessages := make([]interface{}, 0, len(messages)+1)
	newMessages = append(newMessages, messages[:lastUserIdx+1]...)
	newMessages = append(newMessages, toolMessage)
	newMessages = append(newMessages, messages[lastUserIdx+1:]...)

	requestMap["messages"] = newMessages

	// Update context
	updatedBody, err := json.Marshal(requestMap)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	ctx.OriginalRequestBody = updatedBody
	ctx.RAGRetrievedContext = context
	ctx.HasToolsForFactCheck = true
	ctx.ToolResultsContext = context // Store for hallucination detection

	logging.Infof("Injected RAG context as tool role message (%d chars)", len(context))
	return nil
}

// injectAsSystemPrompt injects context into system prompt
func (r *OpenAIRouter) injectAsSystemPrompt(messages []interface{}, context string, requestMap map[string]interface{}, ctx *RequestContext) error {
	// Check if system message exists
	hasSystemMessage := false
	var systemContent string

	if len(messages) > 0 {
		firstMsg, ok := messages[0].(map[string]interface{})
		if ok {
			role, _ := firstMsg["role"].(string)
			if role == "system" {
				hasSystemMessage = true
				systemContent, _ = firstMsg["content"].(string)
			}
		}
	}

	// Prepend context to system message
	contextPrefix := fmt.Sprintf("Context from knowledge base:\n\n%s\n\n", context)

	if hasSystemMessage {
		// Clone the first system message map to avoid mutating shared state
		origFirst, ok := messages[0].(map[string]interface{})
		if !ok {
			return fmt.Errorf("expected first message to be a map when hasSystemMessage is true")
		}
		newFirst := make(map[string]interface{}, len(origFirst)+1)
		for k, v := range origFirst {
			newFirst[k] = v
		}
		newFirst["content"] = contextPrefix + systemContent
		newMessages := make([]interface{}, len(messages))
		copy(newMessages, messages)
		newMessages[0] = newFirst
		requestMap["messages"] = newMessages
	} else {
		systemMessage := map[string]interface{}{
			"role":    "system",
			"content": contextPrefix,
		}
		messages = append([]interface{}{systemMessage}, messages...)
		requestMap["messages"] = messages
	}

	// Update context
	updatedBody, err := json.Marshal(requestMap)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	ctx.OriginalRequestBody = updatedBody
	ctx.RAGRetrievedContext = context
	// Note: For system_prompt mode, we don't set HasToolsForFactCheck
	// as context is in system prompt, not tool messages

	logging.Infof("Injected RAG context into system prompt (%d chars)", len(context))
	return nil
}

// Helper function
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
