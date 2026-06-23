package extproc

import (
	"time"

	"github.com/openai/openai-go"
	"github.com/tidwall/gjson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/inflight"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/latency"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ratelimit"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

type responseUsageMetrics struct {
	promptTokens               int
	cachedPromptTokens         int
	cachedPromptTokensReported bool
	completionTokens           int
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
	cachedPromptTokens := firstExistingGJSON(
		gjson.GetBytes(responseBody, "usage.prompt_tokens_details.cached_tokens"),
		gjson.GetBytes(responseBody, "usage.input_tokens_details.cached_tokens"),
	)
	cachedPromptTokensReported := cachedPromptTokens.Exists()
	if (promptTokens.Exists() && promptTokens.Type != gjson.Number) ||
		(completionTokens.Exists() && completionTokens.Type != gjson.Number) ||
		(cachedPromptTokensReported && cachedPromptTokens.Type != gjson.Number) {
		logging.Errorf("Error parsing tokens from response: usage fields must be numbers")
		metrics.RecordRequestError(model, "parse_error")
		return responseUsageMetrics{}
	}

	return responseUsageMetrics{
		promptTokens:               int(promptTokens.Int()),
		cachedPromptTokens:         clampCachedPromptTokensInt(int(promptTokens.Int()), int(cachedPromptTokens.Int())),
		cachedPromptTokensReported: cachedPromptTokensReported,
		completionTokens:           int(completionTokens.Int()),
	}
}

func firstExistingGJSON(values ...gjson.Result) gjson.Result {
	for _, value := range values {
		if value.Exists() {
			return value
		}
	}
	return gjson.Result{}
}

func (r *OpenAIRouter) reportNonStreamingUsage(
	ctx *RequestContext,
	completionLatency time.Duration,
	usage responseUsageMetrics,
) {
	totalTokens := usage.promptTokens + usage.completionTokens

	if r.RateLimiter != nil && ctx.RateLimitCtx != nil {
		r.RateLimiter.Report(*ctx.RateLimitCtx, ratelimit.TokenUsage{
			InputTokens:  usage.promptTokens,
			OutputTokens: usage.completionTokens,
			TotalTokens:  totalTokens,
		})
	}

	if totalTokens > 0 {
		recordSessionTurn(ctx, usage, r.sessionTurnPricing(ctx.RequestModel))
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
	inflight.End(ctx.RequestModel, ctx.InflightToken)
	ctx.InflightToken = 0

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
	r.observeRouterLearningUsageTelemetry(ctx, completionLatency, usage, replayUsage)
}

func (r *OpenAIRouter) calibrateTokenEstimator(ctx *RequestContext, actualPromptTokens int) {
	if r == nil || r.Classifier == nil || ctx == nil || actualPromptTokens <= 0 {
		return
	}
	byteLen := tokenCalibrationByteLen(ctx)
	if byteLen <= 0 {
		return
	}

	r.Classifier.ObserveTokenUsage("", byteLen, actualPromptTokens)
	if category := tokenCalibrationCategory(ctx); category != "" {
		r.Classifier.ObserveTokenUsage(category, byteLen, actualPromptTokens)
	}
}

func tokenCalibrationByteLen(ctx *RequestContext) int {
	if ctx == nil {
		return 0
	}
	if ctx.VSRContextTextBytes > 0 {
		return ctx.VSRContextTextBytes
	}
	if ctx.RequestQuery != "" {
		return len(ctx.RequestQuery)
	}
	return len(ctx.OriginalRequestBody)
}

func tokenCalibrationCategory(ctx *RequestContext) string {
	if ctx == nil {
		return ""
	}
	if len(ctx.VSRMatchedContext) > 0 {
		return ctx.VSRMatchedContext[0]
	}
	return ctx.VSRSelectedDecisionName
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
		"cached_prompt_tokens":  usage.cachedPromptTokens,
		"completion_tokens":     usage.completionTokens,
		"total_tokens":          totalTokens,
		"completion_latency_ms": completionLatency.Milliseconds(),
	}

	if r.Config != nil {
		pricing, ok := r.Config.GetFullModelPricing(ctx.RequestModel)
		if ok {
			costAmount := costForResponseUsage(usage, pricing)
			currency := pricing.Currency
			metrics.RecordModelCost(ctx.RequestModel, currency, costAmount)
			eventFields["cost"] = costAmount
			eventFields["currency"] = currency
			eventFields["pricing_prompt_per_1m"] = pricing.PromptPer1M
			eventFields["pricing_cached_input_per_1m"] = pricing.CachedInputPer1M
			eventFields["pricing_completion_per_1m"] = pricing.CompletionPer1M
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

func streamingCachedPromptTokens(ctx *RequestContext, promptTokens int) (int, bool) {
	if ctx == nil || ctx.StreamingMetadata == nil {
		return 0, false
	}
	usageMap, ok := ctx.StreamingMetadata["usage"].(map[string]interface{})
	if !ok {
		return 0, false
	}
	for _, key := range []string{"prompt_tokens_details", "input_tokens_details"} {
		details, ok := usageMap[key].(map[string]interface{})
		if !ok {
			continue
		}
		if cached, ok := details["cached_tokens"].(float64); ok {
			return clampCachedPromptTokensInt(promptTokens, int(cached)), true
		}
	}
	return 0, false
}

func recordSessionTurnFromStreamingUsage(
	ctx *RequestContext,
	usage openai.CompletionUsage,
	cachedPromptTokens int,
	cachedPromptTokensReported bool,
	pricing sessiontelemetry.TurnPricing,
) {
	if usage.PromptTokens <= 0 && usage.CompletionTokens <= 0 {
		return
	}
	recordSessionTurn(ctx, responseUsageMetrics{
		promptTokens:               int(usage.PromptTokens),
		cachedPromptTokens:         cachedPromptTokens,
		cachedPromptTokensReported: cachedPromptTokensReported,
		completionTokens:           int(usage.CompletionTokens),
	}, pricing)
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

	cachedPromptTokens, cachedPromptTokensReported := streamingCachedPromptTokens(ctx, int(usage.PromptTokens))
	recordSessionTurnFromStreamingUsage(ctx, usage, cachedPromptTokens, cachedPromptTokensReported, r.sessionTurnPricing(ctx.RequestModel))

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
		promptTokens:               int(usage.PromptTokens),
		cachedPromptTokens:         cachedPromptTokens,
		cachedPromptTokensReported: cachedPromptTokensReported,
		completionTokens:           int(usage.CompletionTokens),
	})
	r.updateRouterReplayUsageCost(ctx, replayUsage)
	r.observeRouterLearningUsageTelemetry(ctx, completionLatency, responseUsageMetrics{
		promptTokens:               int(usage.PromptTokens),
		cachedPromptTokens:         cachedPromptTokens,
		cachedPromptTokensReported: cachedPromptTokensReported,
		completionTokens:           int(usage.CompletionTokens),
	}, replayUsage)
}

func costForResponseUsage(usage responseUsageMetrics, pricing config.ModelPricing) float64 {
	cached := clampCachedPromptTokensInt(usage.promptTokens, usage.cachedPromptTokens)
	uncachedPrompt := usage.promptTokens - cached
	return (float64(uncachedPrompt)*pricing.PromptPer1M +
		float64(cached)*pricing.CachedInputPer1M +
		float64(usage.completionTokens)*pricing.CompletionPer1M) / 1_000_000.0
}

func clampCachedPromptTokensInt(promptTokens, cachedPromptTokens int) int {
	if cachedPromptTokens < 0 {
		return 0
	}
	if cachedPromptTokens > promptTokens {
		return promptTokens
	}
	return cachedPromptTokens
}
