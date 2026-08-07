package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"sort"

	"github.com/openai/openai-go"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
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
	if !r.semanticCacheEnabledForRequest(ctx) {
		return
	}
	if ctx.CacheWriteBypass {
		metrics.RecordCacheWriteSkipped("request_no_store")
		return
	}

	// Status gate: never cache a non-2xx upstream body. Caching an error
	// (4xx/5xx/3xx) would let a later cache hit replay it as an HTTP 200,
	// freezing a transient upstream failure for the whole TTL.
	if skip, reason := shouldSkipCacheWriteForStatus(ctx); skip {
		metrics.RecordCacheWriteSkipped(reason)
		logging.Infof("Skipping cache write for request ID %s: upstream status %d is not 2xx (reason=%s)", ctx.RequestID, ctx.UpstreamStatusCode, reason)
		if span := trace.SpanFromContext(ctx.TraceContext); span.IsRecording() {
			span.SetAttributes(attribute.String(tracing.AttrCacheWriteSkippedReason, reason))
		}
		return
	}

	// Retention drop gate: DSL-author-explicit signal takes precedence over
	// the implicit personalized-context heuristic. Order is locked in
	// docs/design (issue #1747 §2.8): scope -> retention -> personalized -> TTL -> write.
	if skip, reason := ShouldSkipCacheWrite(ctx); skip {
		metrics.RecordCacheWriteSkipped(reason)
		logRetentionSkip(ctx, reason)
		if span := trace.SpanFromContext(ctx.TraceContext); span.IsRecording() {
			span.SetAttributes(attribute.String(tracing.AttrCacheWriteSkippedReason, reason))
		}
		return
	}

	personalized, personalizedReason := ctx.HasPersonalizedContext()
	if personalized && !personalizedCacheWriteAllowed(ctx, personalizedReason) {
		reason := personalizedReason
		metrics.RecordCacheWriteSkipped(reason)
		logging.Infof("Skipping cache write for request ID %s: response has personalized context (reason=%s)", ctx.RequestID, reason)
		if span := trace.SpanFromContext(ctx.TraceContext); span.IsRecording() {
			span.SetAttributes(attribute.String(tracing.AttrCacheWriteSkippedReason, reason))
		}
		return
	}

	ttlSeconds := -1
	if r != nil && r.Config != nil {
		ttlSeconds = r.Config.GetCacheTTLSecondsForDecisionObject(ctx.VSRSelectedDecision)
	}
	ttlSeconds = applyResponseCacheRequestTTL(ctx, ttlSeconds)
	// Retention ttl_turns (when set) overrides the decision/global default,
	// scoping this entry to a turn-bounded lifetime (§2.8 order: ... -> TTL -> write).
	ttlSeconds = applyRetentionTTLOverride(ttlSeconds, ctx)
	if !personalized && semanticCacheWriteAllowed(ctx) {
		if err := r.addSemanticCacheEntry(ctx, responseBody, ttlSeconds); err != nil {
			logging.Errorf("Error adding semantic cache entry: %v", err)
		}
	}
	r.updateExactResponseCache(ctx, responseBody, ttlSeconds)
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

	if !r.semanticCacheEnabledForRequest(ctx) {
		return nil
	}
	if ctx.CacheWriteBypass {
		metrics.RecordCacheWriteSkipped("request_no_store")
		return nil
	}

	// Status gate: never cache a non-2xx upstream body (see updateResponseCache).
	// Streaming errors are usually filtered by the preconditions above, but an
	// error that still carries content must not be frozen into the cache either.
	if skip, reason := shouldSkipCacheWriteForStatus(ctx); skip {
		metrics.RecordCacheWriteSkipped(reason)
		logging.Infof("Skipping streaming cache write for request ID %s: upstream status %d is not 2xx (reason=%s)", ctx.RequestID, ctx.UpstreamStatusCode, reason)
		if span := trace.SpanFromContext(ctx.TraceContext); span.IsRecording() {
			span.SetAttributes(attribute.String(tracing.AttrCacheWriteSkippedReason, reason))
		}
		return nil
	}

	// Retention drop gate runs before any reconstruction work so that we never
	// even build the cache entry for a decision that asked us to skip the
	// write. Same precedence rule as the non-streaming path (§2.8).
	if skip, reason := ShouldSkipCacheWrite(ctx); skip {
		metrics.RecordCacheWriteSkipped(reason)
		logRetentionSkip(ctx, reason)
		if span := trace.SpanFromContext(ctx.TraceContext); span.IsRecording() {
			span.SetAttributes(attribute.String(tracing.AttrCacheWriteSkippedReason, reason))
		}
		return nil
	}

	personalized, personalizedReason := ctx.HasPersonalizedContext()
	if personalized && !personalizedCacheWriteAllowed(ctx, personalizedReason) {
		reason := personalizedReason
		metrics.RecordCacheWriteSkipped(reason)
		logging.Infof("Skipping cache write for streaming request ID %s: response has personalized context (reason=%s)", ctx.RequestID, reason)
		if span := trace.SpanFromContext(ctx.TraceContext); span.IsRecording() {
			span.SetAttributes(attribute.String(tracing.AttrCacheWriteSkippedReason, reason))
		}
		return nil
	}

	usage := extractStreamingUsage(ctx)
	reconstructedJSON, err := buildReconstructedStreamingResponse(ctx, usage, true)
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
	case ctx.StreamingContent == "" &&
		ctx.StreamingReasoning == "" &&
		ctx.StreamingRefusal == "" &&
		len(ctx.StreamingToolCalls) == 0 &&
		!hasReplayableStreamingChoices(ctx):
		logging.Warnf("Streaming response has no replayable payload, skipping cache")
	case ctx.StreamingMetadata["id"] == nil || ctx.StreamingMetadata["model"] == nil:
		logging.Warnf("Streaming response missing required metadata, skipping cache")
	default:
		return nil
	}
	return errSkipStreamingCache
}

func hasReplayableStreamingChoices(ctx *RequestContext) bool {
	if ctx == nil {
		return false
	}
	for _, choice := range ctx.StreamingChoices {
		if choice != nil &&
			(choice.Content != "" ||
				choice.Reasoning != "" ||
				choice.Refusal != "" ||
				len(choice.ToolCalls) > 0) {
			return true
		}
	}
	return false
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
	if len(ctx.StreamingChoices) > 0 {
		return buildIndexedStreamingResponse(ctx, usage, includeToolCalls, id, model, created)
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
	// Preserve reasoning so a cache hit returns the same reasoning_content the
	// live stream delivered (reasoning models stream it under delta.reasoning_content).
	if ctx.StreamingReasoning != "" {
		message["reasoning_content"] = ctx.StreamingReasoning
	}
	if ctx.StreamingRefusal != "" {
		message["refusal"] = ctx.StreamingRefusal
	}

	toolCalls := buildStreamingResponseToolCalls(ctx)
	if includeToolCalls && len(toolCalls) > 0 {
		message["tool_calls"] = toolCalls
		if finishReason == "stop" {
			finishReason = "tool_calls"
		}
	}

	if ctx.StreamingContent == "" &&
		ctx.StreamingReasoning == "" &&
		ctx.StreamingRefusal == "" &&
		(!includeToolCalls || len(toolCalls) == 0) {
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
	return buildStreamingToolCallsFromState(ctx.StreamingToolCalls)
}

func buildStreamingToolCallsFromState(
	state map[int]*StreamingToolCallState,
) []map[string]interface{} {
	indexes := make([]int, 0, len(state))
	for index := range state {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)

	toolCalls := make([]map[string]interface{}, 0, len(indexes))
	for _, index := range indexes {
		call := state[index]
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

func buildIndexedStreamingResponse(
	ctx *RequestContext,
	usage openai.CompletionUsage,
	includeToolCalls bool,
	id string,
	model string,
	created int64,
) ([]byte, error) {
	indexes := make([]int, 0, len(ctx.StreamingChoices))
	for index := range ctx.StreamingChoices {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)
	choices := make([]map[string]interface{}, 0, len(indexes))
	for _, index := range indexes {
		state := ctx.StreamingChoices[index]
		if state == nil {
			return nil, errSkipStreamingCache
		}
		message := map[string]interface{}{"role": "assistant", "content": nil}
		if state.Content != "" {
			message["content"] = state.Content
		}
		if state.Reasoning != "" {
			message["reasoning_content"] = state.Reasoning
		}
		if state.Refusal != "" {
			message["refusal"] = state.Refusal
		}
		toolCalls := buildStreamingToolCallsFromState(state.ToolCalls)
		if includeToolCalls && len(toolCalls) > 0 {
			message["tool_calls"] = toolCalls
		}
		if state.Content == "" && state.Reasoning == "" && state.Refusal == "" &&
			(!includeToolCalls || len(toolCalls) == 0) {
			return nil, errSkipStreamingCache
		}
		finishReason := state.FinishReason
		if finishReason == "" {
			return nil, errSkipStreamingCache
		}
		choices = append(choices, map[string]interface{}{
			"index":         index,
			"message":       message,
			"finish_reason": finishReason,
		})
	}
	return json.Marshal(map[string]interface{}{
		"id":      id,
		"object":  "chat.completion",
		"created": created,
		"model":   model,
		"choices": choices,
		"usage": map[string]interface{}{
			"prompt_tokens":     usage.PromptTokens,
			"completion_tokens": usage.CompletionTokens,
			"total_tokens":      usage.TotalTokens,
		},
	})
}

func (r *OpenAIRouter) cacheReconstructedStreamingResponse(
	ctx *RequestContext,
	reconstructedJSON []byte,
) error {
	if !r.semanticCacheEnabledForRequest(ctx) {
		return nil
	}

	ttlSeconds := -1
	if r != nil && r.Config != nil {
		ttlSeconds = r.Config.GetCacheTTLSecondsForDecisionObject(ctx.VSRSelectedDecision)
	}
	ttlSeconds = applyResponseCacheRequestTTL(ctx, ttlSeconds)
	// Retention ttl_turns (when set) overrides the decision/global default for
	// the reconstructed streaming entry, mirroring the non-streaming path.
	ttlSeconds = applyRetentionTTLOverride(ttlSeconds, ctx)

	if ctx.RequestID == "" {
		logging.Warnf("No request ID available, cannot cache streaming response")
		return nil
	}

	personalized, _ := ctx.HasPersonalizedContext()
	if !personalized && semanticCacheWriteAllowed(ctx) {
		if err := r.addSemanticCacheEntry(ctx, reconstructedJSON, ttlSeconds); err != nil {
			return err
		}
	}
	r.updateExactResponseCache(ctx, reconstructedJSON, ttlSeconds)
	return nil
}

func semanticCacheWriteAllowed(ctx *RequestContext) bool {
	return ctx != nil && (ctx.CacheSemanticSafe || ctx.CacheExactFingerprint == "")
}

func (r *OpenAIRouter) addSemanticCacheEntry(
	ctx *RequestContext,
	responseBody []byte,
	ttlSeconds int,
) error {
	query := cacheQueryForContext(ctx)
	if query == "" {
		return nil
	}
	requestModel := ctx.CacheRequestModel
	if requestModel == "" {
		requestModel = ctx.RequestModel
	}
	identity := ctx.CacheIdentity
	if identity.ExactFingerprint == "" {
		identity = responseCacheIdentity(ctx, requestModel)
	}
	identity.SemanticQuery = query
	writeContext := ctx.TraceContext
	if writeContext == nil {
		writeContext = context.Background()
	}
	service := r.responseCacheService()
	if service == nil {
		return nil
	}
	return service.StoreSemantic(writeContext, cache.CacheWrite{
		Identity:     identity,
		RequestID:    ctx.RequestID,
		RequestBody:  cacheRequestBodyForContext(ctx),
		ResponseBody: responseBody,
		TTL:          cache.TTLPolicyFromLegacySeconds(ttlSeconds),
	})
}
