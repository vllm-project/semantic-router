package extproc

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

func (r *OpenAIRouter) reportCacheHitTelemetry(
	ctx *RequestContext,
	cachedBody []byte,
	lookupLatency time.Duration,
) {
	if r == nil || ctx == nil {
		return
	}
	semanticResponse, err := r.decodeClientResponse(cachedBody, ctx)
	if err != nil {
		logging.ComponentWarnEvent("extproc", "cache_hit_response_decode_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"format":     ctx.SourceFormat,
			"error":      err.Error(),
		})
		return
	}
	usage := responseUsageFromSemanticUsage(semanticResponse.Usage)
	totalTokens := responseUsageTotal(usage)
	if totalTokens > 0 {
		recordSessionTurn(ctx, usage, sessiontelemetry.TurnPricing{})
	}
	if ctx.RequestModel != "" {
		recordModelUsageTokens(ctx.RequestModel, usage)
		metrics.RecordModelCompletionLatency(
			ctx.RequestModel,
			lookupLatency.Seconds(),
		)
	}

	replayUsage := r.buildCacheHitReplayUsage(ctx, usage)
	r.updateRouterReplayUsageCost(ctx, replayUsage)
	r.observeRouterLearningUsageTelemetry(
		ctx,
		lookupLatency,
		usage,
		replayUsage,
	)
	logging.LogEvent("llm_usage", map[string]interface{}{
		"request_id":        ctx.RequestID,
		"model":             ctx.RequestModel,
		"prompt_tokens":     usage.promptTokens,
		"completion_tokens": usage.completionTokens,
		"total_tokens":      totalTokens,
		"cost":              0.0,
		"from_cache":        true,
		"cache_hit":         true,
	})
}

func (r *OpenAIRouter) buildCacheHitReplayUsage(
	ctx *RequestContext,
	usage responseUsageMetrics,
) routerreplay.UsageCost {
	replayUsage := r.buildReplayUsageCost(ctx, usage)
	replayUsage.ActualCost = replayFloat64Ptr(0)
	if replayUsage.BaselineCost != nil {
		replayUsage.CostSavings = replayFloat64Ptr(*replayUsage.BaselineCost)
	}
	return replayUsage
}
