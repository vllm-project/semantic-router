package extproc

import (
	"encoding/json"
	"strings"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/inflight"
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

	responseAPIStream := isResponseAPIRequest(ctx)
	// Envoy body chunks are transport fragments, not SSE frame boundaries.
	// Reassemble every OpenAI-compatible stream before parsing it; otherwise a
	// split `data:` line is forwarded correctly to the client but silently lost
	// from replay, usage reconstruction, and response-side observability.
	parseBody, remainder := reassembleSSEFrames(ctx.PendingSSEBytes, responseBody)
	ctx.PendingSSEBytes = remainder

	chunk := string(parseBody)
	if len(parseBody) > 0 {
		ctx.HasStreamingChunks = true
	}
	r.parseStreamingChunk(chunk, ctx)

	streamDone := strings.Contains(chunk, "data: [DONE]")

	if responseAPIStream {
		bodyMutation := r.buildResponseAPIStreamingBodyMutation(parseBody, ctx)
		if streamDone {
			r.finalizeStreamingResponse(ctx)
		}
		return buildResponseBodyContinueResponse(bodyMutation, responseAPIStreamingHeaderMutation())
	}

	if streamDone {
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
	if ctx.StreamingMetadata == nil {
		ctx.StreamingMetadata = make(map[string]interface{})
	}
	if ctx.StreamingToolCalls == nil {
		ctx.StreamingToolCalls = make(map[int]*StreamingToolCallState)
	}
	if ctx.StreamingChoices == nil {
		ctx.StreamingChoices = make(map[int]*StreamingChoiceState)
	}
	if ctx.ResponseAPIStreamToolCallItemIDs == nil {
		ctx.ResponseAPIStreamToolCallItemIDs = make(map[int]string)
	}
	if ctx.ResponseAPIStreamToolCallOutputIndex == nil {
		ctx.ResponseAPIStreamToolCallOutputIndex = make(map[int]int)
	}
}

func (r *OpenAIRouter) finalizeStreamingResponse(ctx *RequestContext) {
	// Idempotency guard: finalization records completion-latency metrics,
	// ends the inflight token, and attaches the replay body — all of which
	// must happen exactly once. The Anthropic-client streaming cell can
	// reach this twice for one stream (the emitter's streamDone fires on
	// one chunk, then a later terminal chunk still matches the
	// "data: [DONE]" guard), so a second entry must be a no-op.
	if ctx.StreamingComplete {
		return
	}
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

	inflight.End(ctx.RequestModel, ctx.InflightToken)
	ctx.InflightToken = 0

	usage := extractStreamingUsage(ctx)
	r.reportStreamingUsageMetrics(ctx, usage)
	r.calibrateTokenEstimator(ctx, int(usage.PromptTokens))

	if err := r.cacheStreamingResponse(ctx); err != nil {
		logging.Errorf("Failed to cache streaming response: %v", err)
	}

	replayResponseBody, err := buildReconstructedStreamingResponse(ctx, usage, true)
	if err != nil {
		logging.Warnf("Failed to reconstruct streaming replay response body: %v", err)
		replayResponseBody = []byte(ctx.StreamingContent)
	}

	r.attachRouterReplayResponse(ctx, replayResponseBody, true)
	r.maybeStoreResponseAPIStreamingResponse(ctx)
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
		index := streamingChoiceIndex(choice)
		state := streamingChoiceState(ctx, index)
		if delta, ok := choice["delta"].(map[string]interface{}); ok {
			extractStreamingChoiceDelta(state, delta)
			if index == 0 {
				extractStreamingDeltaContent(ctx, delta)
			}
		}
		if finishReason, ok := choice["finish_reason"].(string); ok && finishReason != "" {
			state.FinishReason = finishReason
			if index == 0 {
				ctx.StreamingMetadata["finish_reason"] = finishReason
			}
		}
	}
}

func streamingChoiceIndex(choice map[string]interface{}) int {
	switch value := choice["index"].(type) {
	case float64:
		return int(value)
	case int:
		return value
	default:
		return 0
	}
}

func streamingChoiceState(ctx *RequestContext, index int) *StreamingChoiceState {
	if ctx.StreamingChoices == nil {
		ctx.StreamingChoices = make(map[int]*StreamingChoiceState)
	}
	state := ctx.StreamingChoices[index]
	if state == nil {
		state = &StreamingChoiceState{ToolCalls: make(map[int]*StreamingToolCallState)}
		ctx.StreamingChoices[index] = state
	}
	return state
}

func extractStreamingChoiceDelta(state *StreamingChoiceState, delta map[string]interface{}) {
	if content, ok := delta["content"].(string); ok {
		state.Content += content
	}
	if reasoning, ok := streamingReasoningDelta(delta); ok {
		state.Reasoning += reasoning
	}
	if refusal, ok := delta["refusal"].(string); ok {
		state.Refusal += refusal
	}
}

func streamingReasoningDelta(delta map[string]interface{}) (string, bool) {
	if reasoning, ok := delta["reasoning_content"].(string); ok {
		return reasoning, true
	}
	// Recent OpenAI-compatible reasoning servers use `reasoning` while older
	// ones use `reasoning_content`. Both represent the same neutral streaming
	// field and must produce one canonical accumulator value.
	reasoning, ok := delta["reasoning"].(string)
	return reasoning, ok
}

func extractStreamingDeltaContent(ctx *RequestContext, delta map[string]interface{}) {
	if content, ok := delta["content"].(string); ok && content != "" {
		ctx.StreamingContent += content
	}
	// Accumulate either neutral reasoning alias so replay and cache
	// reconstruction preserve the same field the live stream delivered.
	if reasoning, ok := streamingReasoningDelta(delta); ok && reasoning != "" {
		ctx.StreamingReasoning += reasoning
	}
	if refusal, ok := delta["refusal"].(string); ok && refusal != "" {
		ctx.StreamingRefusal += refusal
	}
}
