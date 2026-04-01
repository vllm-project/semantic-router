package extproc

import (
	"time"

	"github.com/openai/openai-go"
	"github.com/tidwall/gjson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/latency"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ratelimit"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

type responseUsageMetrics struct {
	promptTokens        int
	completionTokens    int
	cachedInputTokens   int
	cacheCreationTokens int
}

// =====================================================================
// NON-STREAMING
// =====================================================================

func parseResponseUsage(responseBody []byte, model string) responseUsageMetrics {
	if !gjson.ValidBytes(responseBody) {
		logging.Errorf("Error parsing tokens from response: invalid JSON")
		metrics.RecordRequestError(model, "parse_error")
		return responseUsageMetrics{}
	}

	promptTokens := gjson.GetBytes(responseBody, "usage.prompt_tokens")
	completionTokens := gjson.GetBytes(responseBody, "usage.completion_tokens")
	if (promptTokens.Exists() && promptTokens.Type != gjson.Number) ||
		(completionTokens.Exists() && completionTokens.Type != gjson.Number) {
		logging.Errorf("Error parsing tokens from response: usage fields must be numbers")
		metrics.RecordRequestError(model, "parse_error")
		return responseUsageMetrics{}
	}

	// Anthropic prompt caching fields (nested under usage.cache_read_input_tokens
	// and usage.cache_creation_input_tokens in the Anthropic response format,
	// or under prompt_tokens_details in the OpenAI format).
	cachedInput := gjson.GetBytes(responseBody, "usage.cache_read_input_tokens")
	if !cachedInput.Exists() {
		cachedInput = gjson.GetBytes(responseBody, "usage.prompt_tokens_details.cached_tokens")
	}
	cacheCreation := gjson.GetBytes(responseBody, "usage.cache_creation_input_tokens")

	return responseUsageMetrics{
		promptTokens:        int(promptTokens.Int()),
		completionTokens:    int(completionTokens.Int()),
		cachedInputTokens:   int(cachedInput.Int()),
		cacheCreationTokens: int(cacheCreation.Int()),
	}
}

func (r *OpenAIRouter) reportNonStreamingUsage(
	ctx *RequestContext,
	completionLatency time.Duration,
	usage responseUsageMetrics,
) {
	totalTokens := usage.promptTokens + usage.completionTokens

	if r.RateLimiter != nil && ctx.RateLimitCtx != nil {
		r.RateLimiter.Report(*ctx.RateLimitCtx, ratelimit.TokenUsage{
			InputTokens:         usage.promptTokens,
			OutputTokens:        usage.completionTokens,
			TotalTokens:         totalTokens,
			CachedInputTokens:   usage.cachedInputTokens,
			CacheCreationTokens: usage.cacheCreationTokens,
		})
	}

	if ctx.RequestModel == "" {
		return
	}

	metrics.RecordModelTokensDetailed(
		ctx.RequestModel,
		float64(usage.promptTokens),
		float64(usage.completionTokens),
	)
	metrics.RecordModelCompletionLatency(ctx.RequestModel, completionLatency.Seconds())

	if usage.completionTokens > 0 {
		timePerToken := completionLatency.Seconds() / float64(usage.completionTokens)
		metrics.RecordModelTPOT(ctx.RequestModel, timePerToken)
		logging.Debugf("Updating TPOT cache for model: %q, TPOT: %.4f", ctx.RequestModel, timePerToken)
		latency.UpdateTPOT(ctx.RequestModel, timePerToken)
	}

	metrics.RecordModelWindowedRequest(
		ctx.RequestModel,
		completionLatency.Seconds(),
		int64(usage.promptTokens),
		int64(usage.completionTokens),
		false,
		false,
	)
	replayUsage := r.recordResponseCost(ctx, completionLatency, usage)
	r.updateRouterReplayUsageCost(ctx, replayUsage)
}

func (r *OpenAIRouter) recordResponseCost(
	ctx *RequestContext,
	completionLatency time.Duration,
	usage responseUsageMetrics,
) routerreplay.UsageCost {
	totalTokens := usage.promptTokens + usage.completionTokens
	replayUsage := r.buildReplayUsageCost(ctx, usage)
	eventFields := map[string]interface{}{
		"request_id":            ctx.RequestID,
		"model":                 ctx.RequestModel,
		"prompt_tokens":         usage.promptTokens,
		"completion_tokens":     usage.completionTokens,
		"total_tokens":          totalTokens,
		"completion_latency_ms": completionLatency.Milliseconds(),
	}

	if r.Config != nil {
		promptRatePer1M, completionRatePer1M, currency, ok := r.Config.GetModelPricing(ctx.RequestModel)
		if ok {
			costAmount := (float64(usage.promptTokens)*promptRatePer1M +
				float64(usage.completionTokens)*completionRatePer1M) / 1_000_000.0
			if currency == "" {
				currency = "USD"
			}
			metrics.RecordModelCost(ctx.RequestModel, currency, costAmount)
			eventFields["cost"] = costAmount
			eventFields["currency"] = currency
			logging.LogEvent("llm_usage", eventFields)
			return replayUsage
		}
	}

	eventFields["cost"] = 0.0
	eventFields["currency"] = "unknown"
	eventFields["pricing"] = "not_configured"
	logging.LogEvent("llm_usage", eventFields)
	return replayUsage
}

// =====================================================================
// STREAMING
// =====================================================================

// streamingCacheUsage holds cache token counts extracted from streaming metadata.
type streamingCacheUsage struct {
	cachedInputTokens   int
	cacheCreationTokens int
}

func extractStreamingUsage(ctx *RequestContext) (openai.CompletionUsage, streamingCacheUsage) {
	usage := openai.CompletionUsage{
		PromptTokens:     0,
		CompletionTokens: 0,
		TotalTokens:      0,
	}
	var cacheUsage streamingCacheUsage
	usageMap, ok := ctx.StreamingMetadata["usage"].(map[string]interface{})
	if !ok {
		return usage, cacheUsage
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
	// Anthropic cache tokens
	if v, ok := usageMap["cache_read_input_tokens"].(float64); ok {
		cacheUsage.cachedInputTokens = int(v)
	}
	if v, ok := usageMap["cache_creation_input_tokens"].(float64); ok {
		cacheUsage.cacheCreationTokens = int(v)
	}
	// OpenAI prompt_tokens_details.cached_tokens
	if details, ok := usageMap["prompt_tokens_details"].(map[string]interface{}); ok {
		if v, ok := details["cached_tokens"].(float64); ok && cacheUsage.cachedInputTokens == 0 {
			cacheUsage.cachedInputTokens = int(v)
		}
	}
	return usage, cacheUsage
}

func (r *OpenAIRouter) reportStreamingUsageMetrics(
	ctx *RequestContext,
	usage openai.CompletionUsage,
	cacheUsage streamingCacheUsage,
) {
	if r.RateLimiter != nil && ctx.RateLimitCtx != nil && (usage.PromptTokens > 0 || usage.CompletionTokens > 0) {
		r.RateLimiter.Report(*ctx.RateLimitCtx, ratelimit.TokenUsage{
			InputTokens:         int(usage.PromptTokens),
			OutputTokens:        int(usage.CompletionTokens),
			TotalTokens:         int(usage.TotalTokens),
			CachedInputTokens:   cacheUsage.cachedInputTokens,
			CacheCreationTokens: cacheUsage.cacheCreationTokens,
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
	logging.ComponentDebugEvent("extproc", "streaming_token_metrics_recorded", map[string]interface{}{
		"model":             ctx.RequestModel,
		"prompt_tokens":     usage.PromptTokens,
		"completion_tokens": usage.CompletionTokens,
	})

	if usage.CompletionTokens > 0 && !ctx.StartTime.IsZero() {
		completionLatency := time.Since(ctx.StartTime).Seconds()
		timePerToken := completionLatency / float64(usage.CompletionTokens)
		metrics.RecordModelTPOT(ctx.RequestModel, timePerToken)
		logging.ComponentDebugEvent("extproc", "streaming_tpot_recorded", map[string]interface{}{
			"model": ctx.RequestModel,
			"tpot":  timePerToken,
		})
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
