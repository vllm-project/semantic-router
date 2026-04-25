package extproc

import (
	"encoding/json"
	"strings"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/latency"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
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
	ctx.CacheWarmthEstimate = latency.EstimateCacheProbability(latency.CacheEstimationInput{
		Model:       ctx.RequestModel,
		TTFTSeconds: ttft,
	})
	maybeEmitTransitionEvent(ctx)
	logging.Debugf("Recorded TTFT on first streamed body chunk: model=%q, TTFT=%.4fs", ctx.RequestModel, ttft)
}

func ensureStreamingState(ctx *RequestContext) {
	if ctx.StreamingChunks == nil {
		ctx.StreamingChunks = make([]string, 0)
	}
	if ctx.StreamingMetadata == nil {
		ctx.StreamingMetadata = make(map[string]interface{})
	}
	if ctx.StreamingToolCalls == nil {
		ctx.StreamingToolCalls = make(map[int]*StreamingToolCallState)
	}
}

func (r *OpenAIRouter) finalizeStreamingResponse(ctx *RequestContext) {
	ctx.StreamingComplete = true
	logging.ComponentDebugEvent("extproc", "streaming_response_finalized", map[string]interface{}{
		"model":            ctx.RequestModel,
		"attempting_cache": true,
	})

	if ctx.RequestModel != "" && !ctx.StartTime.IsZero() {
		completionLatency := time.Since(ctx.StartTime).Seconds()
		metrics.RecordModelCompletionLatency(ctx.RequestModel, completionLatency)
		logging.ComponentDebugEvent("extproc", "streaming_completion_latency_recorded", map[string]interface{}{
			"model":                 ctx.RequestModel,
			"completion_latency_ms": time.Since(ctx.StartTime).Milliseconds(),
		})
	}

	usage, cacheUsage := extractStreamingUsage(ctx)
	r.reportStreamingUsageMetrics(ctx, usage, cacheUsage)

	if err := r.cacheStreamingResponse(ctx); err != nil {
		logging.Errorf("Failed to cache streaming response: %v", err)
	}

	replayResponseBody, err := buildReconstructedStreamingResponse(ctx, usage, true)
	if err != nil {
		logging.Warnf("Failed to reconstruct streaming replay response body: %v", err)
		replayResponseBody = []byte(ctx.StreamingContent)
	}

	r.attachRouterReplayResponse(ctx, replayResponseBody, true)
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
		extractStreamingToolCalls(ctx, chunkData)
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

	for _, rawChoice := range choices {
		choice, ok := rawChoice.(map[string]interface{})
		if !ok {
			continue
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
}
