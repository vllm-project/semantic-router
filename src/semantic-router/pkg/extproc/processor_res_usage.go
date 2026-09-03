package extproc

import (
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/inflight"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/latency"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ratelimit"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

type responseUsageMetrics struct {
	promptTokens               int
	promptTokensReported       bool
	cachedPromptTokens         int
	cachedPromptTokensReported bool
	cacheWriteTokens           int
	cacheWriteTokensReported   bool
	completionTokens           int
	completionTokensReported   bool
	totalTokens                int
	totalTokensReported        bool
	invalid                    bool
	invalidReason              string
}

func responseUsageTotal(usage responseUsageMetrics) int {
	if usage.totalTokens > 0 {
		return usage.totalTokens
	}
	return usage.promptTokens + usage.completionTokens
}

func responseUsageHasPricableBreakdown(usage responseUsageMetrics) bool {
	componentTotal := usage.promptTokens + usage.completionTokens
	componentsReported := usage.promptTokensReported && usage.completionTokensReported
	// Preserve the internal metrics seam for older callers that construct a
	// positive breakdown directly. Wire accounting additionally retains field
	// presence so an explicit 0/0 is authoritative while omitted fields are not.
	if !componentsReported && componentTotal <= 0 {
		return false
	}
	return !usage.totalTokensReported || usage.totalTokens == componentTotal
}

func recordModelUsageTokens(model string, usage responseUsageMetrics) {
	if model == "" {
		return
	}
	if responseUsageHasPricableBreakdown(usage) {
		metrics.RecordModelTokensDetailed(
			model,
			float64(usage.promptTokens),
			float64(usage.completionTokens),
		)
		return
	}
	if totalTokens := responseUsageTotal(usage); totalTokens > 0 {
		metrics.RecordModelTokens(model, float64(totalTokens))
	}
}

// =====================================================================
// NON-STREAMING
// =====================================================================

func (r *OpenAIRouter) reportNonStreamingUsage(
	ctx *RequestContext,
	completionLatency time.Duration,
	usage responseUsageMetrics,
) {
	if usage.invalid {
		usage = responseUsageMetrics{}
	}
	totalTokens := responseUsageTotal(usage)

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

	recordModelUsageTokens(ctx.RequestModel, usage)
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
	if r == nil || ctx == nil || actualPromptTokens <= 0 {
		return
	}
	classifier := r.classifierForRequest(ctx)
	if classifier == nil {
		return
	}
	byteLen := tokenCalibrationByteLen(ctx)
	if byteLen <= 0 {
		return
	}

	classifier.ObserveTokenUsage("", byteLen, actualPromptTokens)
	if category := tokenCalibrationCategory(ctx); category != "" {
		classifier.ObserveTokenUsage(category, byteLen, actualPromptTokens)
	}
	if compressionCategory := contextCompressionTokenCalibrationCategory(ctx); compressionCategory != "" {
		classifier.ObserveTokenUsage(
			compressionCategory,
			len(cacheRequestBodyForContext(ctx)),
			actualPromptTokens,
		)
	}
}

func contextCompressionTokenCalibrationCategory(ctx *RequestContext) string {
	if ctx == nil || ctx.ContextCompressionRevision == "" {
		return ""
	}
	model := strings.TrimSpace(ctx.VSRSelectedModel)
	if model == "" {
		model = strings.TrimSpace(ctx.RequestModel)
	}
	if model == "" {
		return ""
	}
	return "context_compression:" + model
}

func tokenCalibrationByteLen(ctx *RequestContext) int {
	if ctx == nil {
		return 0
	}
	// Structured JSON uses a separate dense-token floor and images use a fixed
	// reserve. Treating either as prose bytes would poison the online 4B/token
	// calibrator with provider-specific schema/vision costs.
	if ctx.VSRContextHasNonText {
		return 0
	}
	if ctx.VSRContextTextBytes > 0 {
		return ctx.VSRContextTextBytes
	}
	if ctx.RequestQuery != "" {
		return len(ctx.RequestQuery)
	}
	return ctx.IngressBodyBytes
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
	totalTokens := responseUsageTotal(usage)
	replayUsage := r.buildReplayUsageCost(ctx, usage)
	eventFields := map[string]interface{}{
		"request_id":            ctx.RequestID,
		"model":                 ctx.RequestModel,
		"prompt_tokens":         usage.promptTokens,
		"cached_prompt_tokens":  usage.cachedPromptTokens,
		"cache_write_tokens":    usage.cacheWriteTokens,
		"completion_tokens":     usage.completionTokens,
		"total_tokens":          totalTokens,
		"completion_latency_ms": completionLatency.Milliseconds(),
	}

	if r.Config != nil {
		pricing, ok := r.Config.GetFullModelPricing(ctx.RequestModel)
		if ok {
			if !responseUsageHasPricableBreakdown(usage) {
				eventFields["pricing"] = "usage_breakdown_unavailable"
				logging.LogEvent("llm_usage", eventFields)
				return replayUsage
			}
			costAmount := costForResponseUsage(usage, pricing)
			currency := pricing.Currency
			metrics.RecordModelCost(ctx.RequestModel, currency, costAmount)
			eventFields["cost"] = costAmount
			eventFields["currency"] = currency
			eventFields["pricing_prompt_per_1m"] = pricing.PromptPer1M
			eventFields["pricing_cached_input_per_1m"] = pricing.CachedInputPer1M
			eventFields["pricing_cache_write_per_1m"] = effectiveCacheWriteRate(pricing)
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

func clampCachedPromptTokensInt(promptTokens, cachedPromptTokens int) int {
	if cachedPromptTokens < 0 {
		return 0
	}
	if cachedPromptTokens > promptTokens {
		return promptTokens
	}
	return cachedPromptTokens
}
