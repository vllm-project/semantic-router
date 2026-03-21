package extproc

import (
	"encoding/json"
	"errors"
	"strings"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/latency"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ratelimit"
)

func (r *OpenAIRouter) handleStreamingResponseBody(
	responseBody []byte,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	recordStreamingTTFT(ctx)
	ensureStreamingState(ctx)

	chunk := string(responseBody)
	ctx.StreamingChunks = append(ctx.StreamingChunks, chunk)
	r.parseStreamingChunk(chunk, ctx)

	if strings.Contains(chunk, "data: [DONE]") {
		r.finalizeStreamingResponse(ctx)
	}

	return buildResponseBodyContinueResponse(nil, nil)
}

func recordStreamingTTFT(ctx *RequestContext) {
	if ctx == nil || ctx.TTFTRecorded || ctx.ProcessingStartTime.IsZero() || ctx.RequestModel == "" {
		return
	}

	ttft := time.Since(ctx.ProcessingStartTime).Seconds()
	if ttft <= 0 {
		return
	}

	metrics.RecordModelTTFT(ctx.RequestModel, ttft)
	ctx.TTFTSeconds = ttft
	ctx.TTFTRecorded = true
	latency.UpdateTTFT(ctx.RequestModel, ttft)
	logging.Debugf("Recorded TTFT on first streamed body chunk: model=%q, TTFT=%.4fs", ctx.RequestModel, ttft)
}

func ensureStreamingState(ctx *RequestContext) {
	if ctx.StreamingChunks == nil {
		ctx.StreamingChunks = make([]string, 0)
	}
	if ctx.StreamingMetadata == nil {
		ctx.StreamingMetadata = make(map[string]interface{})
	}
}

func (r *OpenAIRouter) finalizeStreamingResponse(ctx *RequestContext) {
	ctx.StreamingComplete = true
	logging.Infof("Streaming response completed, attempting to cache")

	if ctx.RequestModel != "" && !ctx.StartTime.IsZero() {
		completionLatency := time.Since(ctx.StartTime).Seconds()
		metrics.RecordModelCompletionLatency(ctx.RequestModel, completionLatency)
		logging.Infof(
			"Recorded completion latency for streaming response: model=%s, latency=%.3fs",
			ctx.RequestModel,
			completionLatency,
		)
	}

	if err := r.cacheStreamingResponse(ctx); err != nil {
		logging.Errorf("Failed to cache streaming response: %v", err)
	}

	r.attachRouterReplayResponse(ctx, []byte(ctx.StreamingContent), true)
}

// parseStreamingChunk parses an SSE chunk to extract content and metadata.
func (r *OpenAIRouter) parseStreamingChunk(chunk string, ctx *RequestContext) {
	lines := strings.Split(chunk, "\n")
	for _, line := range lines {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimSpace(strings.TrimPrefix(line, "data: "))
		if data == "[DONE]" {
			continue
		}

		var chunkData map[string]interface{}
		if err := json.Unmarshal([]byte(data), &chunkData); err != nil {
			continue
		}

		extractStreamingMetadata(ctx, chunkData)
		extractStreamingContent(ctx, chunkData)
		if usage, ok := chunkData["usage"].(map[string]interface{}); ok {
			ctx.StreamingMetadata["usage"] = usage
		}
	}
}

func extractStreamingMetadata(ctx *RequestContext, chunkData map[string]interface{}) {
	if ctx.StreamingMetadata["id"] != nil {
		return
	}

	if id, ok := chunkData["id"].(string); ok {
		ctx.StreamingMetadata["id"] = id
	}
	if model, ok := chunkData["model"].(string); ok {
		ctx.StreamingMetadata["model"] = model
	}
	if created, ok := chunkData["created"].(float64); ok {
		ctx.StreamingMetadata["created"] = int64(created)
	}
}

func extractStreamingContent(ctx *RequestContext, chunkData map[string]interface{}) {
	choices, ok := chunkData["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return
	}
	if delta, ok := choice["delta"].(map[string]interface{}); ok {
		if content, ok := delta["content"].(string); ok && content != "" {
			ctx.StreamingContent += content
		}
	}
	if finishReason, ok := choice["finish_reason"].(string); ok && finishReason != "" {
		ctx.StreamingMetadata["finish_reason"] = finishReason
	}
}

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

func extractStreamingUsage(ctx *RequestContext) openai.CompletionUsage {
	usage := openai.CompletionUsage{
		PromptTokens:     0,
		CompletionTokens: 0,
		TotalTokens:      0,
	}
	usageMap, ok := ctx.StreamingMetadata["usage"].(map[string]interface{})
	if !ok {
		return usage
	}

	if promptTokens, ok := usageMap["prompt_tokens"].(float64); ok {
		usage.PromptTokens = int64(promptTokens)
	}
	if completionTokens, ok := usageMap["completion_tokens"].(float64); ok {
		usage.CompletionTokens = int64(completionTokens)
	}
	if totalTokens, ok := usageMap["total_tokens"].(float64); ok {
		usage.TotalTokens = int64(totalTokens)
	}
	return usage
}

func (r *OpenAIRouter) reportStreamingUsageMetrics(
	ctx *RequestContext,
	usage openai.CompletionUsage,
) {
	if r.RateLimiter != nil && ctx.RateLimitCtx != nil && (usage.PromptTokens > 0 || usage.CompletionTokens > 0) {
		r.RateLimiter.Report(*ctx.RateLimitCtx, ratelimit.TokenUsage{
			InputTokens:  int(usage.PromptTokens),
			OutputTokens: int(usage.CompletionTokens),
			TotalTokens:  int(usage.TotalTokens),
		})
	}

	if ctx.RequestModel == "" || (usage.PromptTokens == 0 && usage.CompletionTokens == 0) {
		return
	}

	metrics.RecordModelTokensDetailed(
		ctx.RequestModel,
		float64(usage.PromptTokens),
		float64(usage.CompletionTokens),
	)
	logging.Infof(
		"Recorded token metrics for streaming response: model=%s, prompt=%d, completion=%d",
		ctx.RequestModel,
		usage.PromptTokens,
		usage.CompletionTokens,
	)

	if usage.CompletionTokens > 0 && !ctx.StartTime.IsZero() {
		completionLatency := time.Since(ctx.StartTime).Seconds()
		timePerToken := completionLatency / float64(usage.CompletionTokens)
		metrics.RecordModelTPOT(ctx.RequestModel, timePerToken)
		logging.Infof("Recorded TPOT for streaming response: model=%s, TPOT=%.4f", ctx.RequestModel, timePerToken)
		latency.UpdateTPOT(ctx.RequestModel, timePerToken)
	}

	completionLatency := time.Duration(0)
	if !ctx.StartTime.IsZero() {
		completionLatency = time.Since(ctx.StartTime)
	}
	replayUsage := r.recordResponseCost(ctx, completionLatency, responseUsageMetrics{
		promptTokens:     int(usage.PromptTokens),
		completionTokens: int(usage.CompletionTokens),
	})
	r.updateRouterReplayUsageCost(ctx, replayUsage)
}

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
	// Skip cache store if a decision was selected but doesn't have semantic-cache enabled
	if ctx.VSRSelectedDecisionName != "" && r.Config != nil &&
		!r.Config.IsCacheEnabledForDecision(ctx.VSRSelectedDecisionName) {
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
