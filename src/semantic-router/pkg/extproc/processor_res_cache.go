package extproc

import (
	"encoding/json"
	"errors"

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
	if !r.semanticCacheEnabledForScope(ctx.VSRSelectedDecisionName) {
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
		ttlSeconds = r.Config.GetCacheTTLSecondsForDecision(ctx.VSRSelectedDecisionName)
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

	usage := extractStreamingUsage(ctx)
	r.reportStreamingUsageMetrics(ctx, usage)

	reconstructedJSON, err := buildReconstructedStreamingResponse(ctx, usage)
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
) ([]byte, error) {
	finishReason := "stop"
	if finishReasonValue, ok := ctx.StreamingMetadata["finish_reason"].(string); ok && finishReasonValue != "" {
		finishReason = finishReasonValue
	}

	reconstructed := openai.ChatCompletion{
		ID:      ctx.StreamingMetadata["id"].(string),
		Object:  "chat.completion",
		Created: ctx.StreamingMetadata["created"].(int64),
		Model:   ctx.StreamingMetadata["model"].(string),
		Choices: []openai.ChatCompletionChoice{
			{
				Index: 0,
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: ctx.StreamingContent,
				},
				FinishReason: finishReason,
			},
		},
		Usage: usage,
	}

	if len(reconstructed.Choices) == 0 || reconstructed.Choices[0].Message.Content == "" {
		logging.Warnf("Reconstructed response has no valid choices or content, skipping cache")
		return nil, errSkipStreamingCache
	}

	reconstructedJSON, err := json.Marshal(reconstructed)
	if err != nil {
		logging.Errorf("Failed to marshal reconstructed response: %v", err)
		return nil, err
	}
	return reconstructedJSON, nil
}

func (r *OpenAIRouter) cacheReconstructedStreamingResponse(
	ctx *RequestContext,
	reconstructedJSON []byte,
) error {
	if !r.semanticCacheEnabledForScope(ctx.VSRSelectedDecisionName) {
		return nil
	}

	ttlSeconds := -1
	if r != nil && r.Config != nil {
		ttlSeconds = r.Config.GetCacheTTLSecondsForDecision(ctx.VSRSelectedDecisionName)
	}

	if ctx.RequestID == "" {
		logging.Warnf("No request ID available, cannot cache streaming response")
		return nil
	}

	if ctx.RequestQuery == "" || ctx.RequestModel == "" {
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
		ctx.RequestQuery,
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
		logging.Errorf("Error caching streaming response with UpdateWithResponse: %v", err)
		return err
	}
	logging.Infof("Successfully cached streaming response (via UpdateWithResponse) for request ID: %s", requestID)
	return nil
}
