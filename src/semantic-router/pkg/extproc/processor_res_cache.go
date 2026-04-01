package extproc

import (
	"encoding/json"
	"errors"
	"sort"

	"github.com/openai/openai-go"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
)

// =====================================================================
// NON-STREAMING
// =====================================================================

func (r *OpenAIRouter) updateResponseCache(ctx *RequestContext, responseBody []byte) {
	if ctx.RequestID == "" || responseBody == nil {
		return
	}
	decisionName := ctx.VSRSelectedDecisionName
	if !r.semanticCacheEnabledForScope(decisionName) {
		return
	}

	if ok, reason := ctx.HasPersonalizedContext(); ok {
		metrics.RecordCacheWriteSkipped(reason)
		logging.Infof("Skipping cache write for request ID %s: response has personalized context (reason=%s)", ctx.RequestID, reason)
		if span := trace.SpanFromContext(ctx.TraceContext); span.IsRecording() {
			span.SetAttributes(attribute.String(tracing.AttrCacheWriteSkippedReason, reason))
		}
		return
	}

	ttlSeconds := -1
	if r != nil && r.Config != nil {
		ttlSeconds = r.Config.GetCacheTTLSecondsForDecision(decisionName)
	}
	if err := r.Cache.UpdateWithResponse(ctx.RequestID, responseBody, ttlSeconds); err != nil {
		logging.Errorf("Error updating cache: %v", err)
		return
	}
	logging.Infof("Cache updated for request ID: %s", ctx.RequestID)
}

// =====================================================================
// STREAMING
// =====================================================================

// cacheStreamingResponse reconstructs a ChatCompletion from accumulated chunks and caches it.
func (r *OpenAIRouter) cacheStreamingResponse(ctx *RequestContext) error {
	if err := validateStreamingCachePreconditions(ctx); err != nil {
		return nil
	}

	if ok, reason := ctx.HasPersonalizedContext(); ok {
		metrics.RecordCacheWriteSkipped(reason)
		logging.Infof("Skipping cache write for streaming request ID %s: response has personalized context (reason=%s)", ctx.RequestID, reason)
		if span := trace.SpanFromContext(ctx.TraceContext); span.IsRecording() {
			span.SetAttributes(attribute.String(tracing.AttrCacheWriteSkippedReason, reason))
		}
		return nil
	}

	usage, cacheUsage := extractStreamingUsage(ctx)
	r.reportStreamingUsageMetrics(ctx, usage, cacheUsage)

	reconstructedJSON, err := buildReconstructedStreamingResponse(ctx, usage, false)
	if err != nil {
		if errors.Is(err, errSkipStreamingCache) {
			return nil
		}
		return err
	}

	return r.cacheReconstructedStreamingResponse(ctx, reconstructedJSON)
}

func validateStreamingCachePreconditions(ctx *RequestContext) error {
	switch {
	case !ctx.StreamingComplete:
		logging.Warnf("Stream not completed (no [DONE] marker), skipping cache")
	case ctx.StreamingAborted:
		logging.Warnf("Stream was aborted, skipping cache")
	case ctx.StreamingContent == "":
		logging.Warnf("Streaming response has no content, skipping cache")
	case ctx.StreamingMetadata["id"] == nil || ctx.StreamingMetadata["model"] == nil:
		logging.Warnf("Streaming response missing required metadata, skipping cache")
	default:
		return nil
	}
	return errSkipStreamingCache
}

var errSkipStreamingCache = &streamingCacheSkipError{}

type streamingCacheSkipError struct{}

func (e *streamingCacheSkipError) Error() string { return "skip streaming cache" }

func buildReconstructedStreamingResponse(
	ctx *RequestContext,
	usage openai.CompletionUsage,
	includeToolCalls bool,
) ([]byte, error) {
	if ctx == nil {
		return nil, errSkipStreamingCache
	}

	id, idOK := ctx.StreamingMetadata["id"].(string)
	model, modelOK := ctx.StreamingMetadata["model"].(string)
	created, createdOK := ctx.StreamingMetadata["created"].(int64)
	if !idOK || !modelOK || !createdOK {
		logging.Warnf("Streaming response missing metadata required for reconstruction, skipping")
		return nil, errSkipStreamingCache
	}

	finishReason := "stop"
	if finishReasonValue, ok := ctx.StreamingMetadata["finish_reason"].(string); ok && finishReasonValue != "" {
		finishReason = finishReasonValue
	}

	message := map[string]interface{}{
		"role": "assistant",
	}
	if ctx.StreamingContent != "" {
		message["content"] = ctx.StreamingContent
	} else {
		message["content"] = nil
	}

	toolCalls := buildStreamingResponseToolCalls(ctx)
	if includeToolCalls && len(toolCalls) > 0 {
		message["tool_calls"] = toolCalls
		if finishReason == "stop" {
			finishReason = "tool_calls"
		}
	}

	if ctx.StreamingContent == "" && (!includeToolCalls || len(toolCalls) == 0) {
		logging.Warnf("Reconstructed response has no valid assistant payload, skipping cache")
		return nil, errSkipStreamingCache
	}

	reconstructed := map[string]interface{}{
		"id":      id,
		"object":  "chat.completion",
		"created": created,
		"model":   model,
		"choices": []map[string]interface{}{
			{
				"index":         0,
				"message":       message,
				"finish_reason": finishReason,
			},
		},
		"usage": map[string]interface{}{
			"prompt_tokens":     usage.PromptTokens,
			"completion_tokens": usage.CompletionTokens,
			"total_tokens":      usage.TotalTokens,
		},
	}

	reconstructedJSON, err := json.Marshal(reconstructed)
	if err != nil {
		logging.Errorf("Failed to marshal reconstructed response: %v", err)
		return nil, err
	}
	return reconstructedJSON, nil
}

func buildStreamingResponseToolCalls(ctx *RequestContext) []map[string]interface{} {
	if ctx == nil || len(ctx.StreamingToolCalls) == 0 {
		return nil
	}

	indexes := make([]int, 0, len(ctx.StreamingToolCalls))
	for index := range ctx.StreamingToolCalls {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)

	toolCalls := make([]map[string]interface{}, 0, len(indexes))
	for _, index := range indexes {
		call := ctx.StreamingToolCalls[index]
		if call == nil || (call.ID == "" && call.Name == "" && call.Arguments == "") {
			continue
		}
		toolCalls = append(toolCalls, map[string]interface{}{
			"id":   call.ID,
			"type": "function",
			"function": map[string]interface{}{
				"name":      call.Name,
				"arguments": call.Arguments,
			},
		})
	}
	return toolCalls
}

func (r *OpenAIRouter) cacheReconstructedStreamingResponse(
	ctx *RequestContext,
	reconstructedJSON []byte,
) error {
	decisionName := ctx.VSRSelectedDecisionName
	if !r.semanticCacheEnabledForScope(decisionName) {
		return nil
	}

	ttlSeconds := -1
	if r != nil && r.Config != nil {
		ttlSeconds = r.Config.GetCacheTTLSecondsForDecision(decisionName)
	}

	if ctx.RequestID == "" {
		logging.Warnf("No request ID available, cannot cache streaming response")
		return nil
	}

	if cacheQueryForContext(ctx) == "" || ctx.RequestModel == "" {
		return r.updateStreamingCacheEntry(ctx.RequestID, reconstructedJSON, ttlSeconds)
	}

	if err := r.addStreamingCacheEntry(ctx, reconstructedJSON, ttlSeconds); err != nil {
		logging.Errorf("Error caching streaming response with AddEntry: %v", err)
		return r.updateStreamingCacheEntry(ctx.RequestID, reconstructedJSON, ttlSeconds)
	}

	logging.Infof("Successfully cached streaming response (via AddEntry) for request ID: %s", ctx.RequestID)
	return nil
}

func (r *OpenAIRouter) addStreamingCacheEntry(
	ctx *RequestContext,
	reconstructedJSON []byte,
	ttlSeconds int,
) error {
	return r.Cache.AddEntry(
		ctx.RequestID,
		ctx.RequestModel,
		cacheQueryForContext(ctx),
		streamingCacheRequestBody(ctx),
		reconstructedJSON,
		ttlSeconds,
	)
}

func streamingCacheRequestBody(ctx *RequestContext) []byte {
	if ctx.OriginalRequestBody == nil {
		return []byte("{}")
	}
	return ctx.OriginalRequestBody
}

func (r *OpenAIRouter) updateStreamingCacheEntry(
	requestID string,
	reconstructedJSON []byte,
	ttlSeconds int,
) error {
	if err := r.Cache.UpdateWithResponse(requestID, reconstructedJSON, ttlSeconds); err != nil {
		logging.Errorf("Cache update failed for streaming %s: %v", requestID, err)
		return err
	}
	return nil
}
