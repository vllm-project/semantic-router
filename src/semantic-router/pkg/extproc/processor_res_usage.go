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
	cacheWrite5mTokens         int
	cacheWrite5mTokensReported bool
	cacheWrite1hTokens         int
	cacheWrite1hTokensReported bool
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
	recordSessionTurnOutcome(ctx, usage, r.sessionTurnPricing(ctx.RequestModel))
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
	inflight.End(ctx.InflightModel, ctx.InflightToken)
	ctx.InflightToken = 0

	if usage.completionTokens > 0 {
		timePerToken := completionLatency.Seconds() / float64(usage.completionTokens)
		metrics.RecordModelResponseDurationPerOutputToken(ctx.RequestModel, timePerToken)
		logging.Debugf("Updating TPOT cache for model: %q, TPOT: %.4f", ctx.RequestModel, timePerToken)
		latency.UpdateTPOT(ctx.RequestModel, timePerToken)
	}

	metrics.RecordModelWindowedRequest(
		ctx.RequestModel,
		completionLatency.Seconds(),
		int64(usage.promptTokens),
		int64(usage.completionTokens),
	)
	replayUsage := r.recordResponseCost(ctx, completionLatency, usage)
	r.settleRequestSpend(ctx, &replayUsage)
	r.updateRouterReplayUsageCost(ctx, replayUsage)
	r.observeRouterLearningUsageTelemetry(ctx, completionLatency, usage, replayUsage)
}

func (r *OpenAIRouter) calibrateTokenEstimator(ctx *RequestContext, actualPromptTokens int) {
	if r == nil || ctx == nil {
		return
	}
	r.calibrateTokenEstimatorForModel(
		ctx,
		calibrationTargetModel(ctx),
		tokenCalibrationByteLen(ctx),
		actualPromptTokens,
	)
}

func (r *OpenAIRouter) calibrateTokenEstimatorForModel(
	ctx *RequestContext,
	targetModel string,
	byteLen int,
	actualPromptTokens int,
) {
	if r == nil || ctx == nil || actualPromptTokens <= 0 || byteLen <= 0 {
		return
	}
	model := strings.TrimSpace(targetModel)
	if ctx.LooperStreamingActive && model == "" {
		return
	}
	if model != "" && r.isVirtualCalibrationModel(model, ctx) {
		return
	}
	classifier := r.classifierForRequest(ctx)
	if classifier == nil {
		return
	}

	classifier.ObserveTokenUsage("", byteLen, actualPromptTokens)
	if model != "" {
		classifier.ObserveTokenUsage("model:"+model, byteLen, actualPromptTokens)
	}
	if category := tokenCalibrationCategory(ctx); category != "" {
		classifier.ObserveTokenUsage(category, byteLen, actualPromptTokens)
	}
	if compressionCategory := contextCompressionTokenCalibrationCategoryForModel(ctx, model); compressionCategory != "" {
		classifier.ObserveTokenUsage(
			compressionCategory,
			len(cacheRequestBodyForContext(ctx)),
			actualPromptTokens,
		)
	}
}

func calibrationTargetModel(ctx *RequestContext) string {
	if ctx == nil {
		return ""
	}
	if ctx.LooperStreamingActive {
		return firstNonEmpty(strings.TrimSpace(ctx.LooperTargetModel), strings.TrimSpace(ctx.VSRSelectedModel))
	}
	return firstNonEmpty(strings.TrimSpace(ctx.VSRSelectedModel), strings.TrimSpace(ctx.RequestModel))
}

func (r *OpenAIRouter) isVirtualCalibrationModel(model string, ctx *RequestContext) bool {
	trimmed := strings.TrimSpace(model)
	if trimmed == "" {
		return true
	}
	if strings.EqualFold(trimmed, "auto") {
		return true
	}
	if r != nil && r.Config != nil && r.Config.IsEntrypointModelName(trimmed) {
		return true
	}
	if ctx != nil && ctx.LooperStreamingActive {
		if ctx.LooperTargetModel != "" && trimmed != strings.TrimSpace(ctx.LooperTargetModel) {
			return true
		}
		if strings.TrimSpace(ctx.RequestModel) != "" &&
			trimmed == strings.TrimSpace(ctx.RequestModel) &&
			trimmed != strings.TrimSpace(ctx.LooperTargetModel) &&
			trimmed != strings.TrimSpace(ctx.VSRSelectedModel) {
			return true
		}
	}
	return false
}

func contextCompressionTokenCalibrationCategory(ctx *RequestContext) string {
	return contextCompressionTokenCalibrationCategoryForModel(ctx, calibrationTargetModel(ctx))
}

func contextCompressionTokenCalibrationCategoryForModel(ctx *RequestContext, targetModel string) string {
	if ctx == nil || ctx.ContextCompressionRevision == "" {
		return ""
	}
	model := strings.TrimSpace(targetModel)
	if model == "" {
		model = strings.TrimSpace(ctx.VSRSelectedModel)
	}
	if model == "" {
		model = strings.TrimSpace(ctx.RequestModel)
	}
	if model == "" || strings.EqualFold(model, "auto") {
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
	if ctx.LooperStreamingActive {
		eventFields["target_model"] = firstNonEmpty(ctx.LooperTargetModel, ctx.VSRSelectedModel)
		eventFields["intermediate_prompt_tokens"] = ctx.LooperIntermediateUsage.PromptTokens
		eventFields["intermediate_completion_tokens"] = ctx.LooperIntermediateUsage.CompletionTokens
		if len(ctx.LooperModelsUsed) > 0 {
			eventFields["looper_models_used"] = ctx.LooperModelsUsed
		}
		if ctx.LooperIterations > 0 {
			eventFields["looper_iterations"] = ctx.LooperIterations
		}
		eventFields["looper_trace_recorded"] = ctx.LooperExecutionTrace != nil
		eventFields["cost"] = 0.0
		eventFields["currency"] = "unknown"
		eventFields["pricing"] = "looper_cascade_unpriced"
		logging.LogEvent("llm_usage", eventFields)
		return replayUsage
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
			ctx.RequestCost = costAmount
			ctx.RequestCostCurrency = currency
			ctx.RequestCostPriced = true
			eventFields["cost"] = costAmount
			eventFields["currency"] = currency
			eventFields["pricing_prompt_per_1m"] = pricing.PromptPer1M
			eventFields["pricing_cached_input_per_1m"] = pricing.CachedInputPer1M
			eventFields["pricing_cache_write_per_1m"] = effectiveCacheWriteRate(pricing)
			eventFields["pricing_cache_write_1h_per_1m"] = effectiveCacheWrite1hRate(pricing)
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
