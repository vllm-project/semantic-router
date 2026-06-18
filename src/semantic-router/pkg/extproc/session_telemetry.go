package extproc

import (
	"math"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/consts"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

const routerCacheIdleBoundarySeconds = 300.0

type routerCacheAccounting struct {
	source                string
	estimatedCachedTokens int
	estimatedSavings      float64
	confidence            float64
}

// sessionTurnPricing looks up the active pricing for model from the router config
// and converts it to the sessiontelemetry value type.
func (r *OpenAIRouter) sessionTurnPricing(model string) sessiontelemetry.TurnPricing {
	if r.Config == nil {
		return sessiontelemetry.TurnPricing{}
	}
	p, ok := r.Config.GetFullModelPricing(model)
	if !ok {
		return sessiontelemetry.TurnPricing{}
	}
	return sessiontelemetry.TurnPricing{
		Currency:         p.Currency,
		PromptPer1M:      p.PromptPer1M,
		CompletionPer1M:  p.CompletionPer1M,
		CachedInputPer1M: p.CachedInputPer1M,
	}
}

func recordSessionTurn(ctx *RequestContext, usage responseUsageMetrics, pricing sessiontelemetry.TurnPricing) {
	if ctx == nil || usage.promptTokens+usage.completionTokens <= 0 {
		return
	}
	sessiontelemetry.RecordLastModel(ctx.SessionID, ctx.RequestModel)
	accounting := estimateRouterCacheAccounting(ctx, usage, pricing)

	domain := consts.UnknownLabel
	if ctx.VSRSelectedCategory != "" {
		domain = ctx.VSRSelectedCategory
	}
	p := sessiontelemetry.TurnParams{
		RequestID:                   ctx.RequestID,
		Model:                       ctx.RequestModel,
		Domain:                      domain,
		PromptTokens:                usage.promptTokens,
		CachedPromptTokens:          usage.cachedPromptTokens,
		EstimatedCachedPromptTokens: accounting.estimatedCachedTokens,
		CompletionTokens:            usage.completionTokens,
		EstimatedCacheSavings:       accounting.estimatedSavings,
		CacheAccountingSource:       accounting.source,
		CacheAccountingConfidence:   accounting.confidence,
		Pricing:                     pricing,
	}
	if ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.IsResponseAPIRequest {
		if ctx.ResponseAPICtx.ConversationID == "" {
			return
		}
		p.ResponseAPI = &sessiontelemetry.ResponseAPIInput{
			ConversationID: ctx.ResponseAPICtx.ConversationID,
			HistoryLen:     len(ctx.ResponseAPICtx.ConversationHistory),
		}
	} else {
		userID := extractUserID(ctx)
		if userID == "" || len(ctx.ChatCompletionMessages) == 0 {
			recordRouterSessionUsageFromContext(ctx, usage, pricing, accounting)
			return
		}
		msgs := make([]sessiontelemetry.ChatMessage, len(ctx.ChatCompletionMessages))
		for i := range ctx.ChatCompletionMessages {
			msgs[i] = sessiontelemetry.ChatMessage{
				Role:    ctx.ChatCompletionMessages[i].Role,
				Content: ctx.ChatCompletionMessages[i].Content,
			}
		}
		p.Chat = &sessiontelemetry.ChatInput{UserID: userID, Messages: msgs}
	}
	sessiontelemetry.RecordTurn(p)
	recordRouterLearningUsageFromContext(ctx, usage, pricing, accounting)
}

func recordRouterSessionUsageFromContext(
	ctx *RequestContext,
	usage responseUsageMetrics,
	pricing sessiontelemetry.TurnPricing,
	accounting routerCacheAccounting,
) {
	if ctx == nil || ctx.SessionID == "" || ctx.RequestModel == "" {
		recordRouterLearningUsageFromContext(ctx, usage, pricing, accounting)
		return
	}
	sessiontelemetry.RecordSessionUsage(sessiontelemetry.SessionUsageParams{
		SessionID:                   ctx.SessionID,
		Model:                       ctx.RequestModel,
		PromptTokens:                usage.promptTokens,
		CachedPromptTokens:          usage.cachedPromptTokens,
		EstimatedCachedPromptTokens: accounting.estimatedCachedTokens,
		CompletionTokens:            usage.completionTokens,
		Cost:                        sessionTurnCost(usage, pricing),
		EstimatedCacheSavings:       accounting.estimatedSavings,
		CacheAccountingSource:       accounting.source,
		Timestamp:                   time.Now(),
	})
	recordRouterLearningUsageFromContext(ctx, usage, pricing, accounting)
}

func recordRouterLearningUsageFromContext(
	ctx *RequestContext,
	usage responseUsageMetrics,
	pricing sessiontelemetry.TurnPricing,
	accounting routerCacheAccounting,
) {
	if ctx == nil || ctx.VSRLearningSessionID == "" || ctx.VSRLearningSessionID == ctx.SessionID || ctx.RequestModel == "" {
		return
	}
	sessiontelemetry.RecordSessionUsage(sessiontelemetry.SessionUsageParams{
		SessionID:                   ctx.VSRLearningSessionID,
		Model:                       ctx.RequestModel,
		PromptTokens:                usage.promptTokens,
		CachedPromptTokens:          usage.cachedPromptTokens,
		EstimatedCachedPromptTokens: accounting.estimatedCachedTokens,
		CompletionTokens:            usage.completionTokens,
		Cost:                        sessionTurnCost(usage, pricing),
		EstimatedCacheSavings:       accounting.estimatedSavings,
		CacheAccountingSource:       accounting.source,
		Timestamp:                   time.Now(),
	})
}

func estimateRouterCacheAccounting(
	ctx *RequestContext,
	usage responseUsageMetrics,
	pricing sessiontelemetry.TurnPricing,
) routerCacheAccounting {
	if usage.promptTokens <= 0 {
		return routerCacheAccounting{source: "no_prompt_tokens"}
	}
	if usage.cachedPromptTokensReported {
		return routerCacheAccounting{source: "backend_reported", confidence: 1.0}
	}
	if ctx == nil || ctx.SessionID == "" {
		return routerCacheAccounting{source: "missing_session"}
	}
	if ctx.RequestModel == "" {
		return routerCacheAccounting{source: "missing_request_model"}
	}
	if ctx.PreviousModel == "" {
		return routerCacheAccounting{source: "missing_previous_model"}
	}
	if ctx.PreviousModel != ctx.RequestModel {
		return routerCacheAccounting{source: "switch_checkout", confidence: 1.0}
	}
	if ctx.SessionIdleKnown && ctx.SessionIdleSeconds >= routerCacheIdleBoundarySeconds {
		return routerCacheAccounting{source: "idle_boundary", confidence: 0.8}
	}
	if ctx.HistoryTokenCount <= 0 {
		return routerCacheAccounting{source: "no_history"}
	}

	warmth := ctx.CacheWarmthEstimate
	confidence := 0.65
	if warmth <= 0 {
		warmth = 0.5
		confidence = 0.35
	}
	warmth = clampFloat64(warmth, 0, 1)
	reusablePrompt := minInt(usage.promptTokens, ctx.HistoryTokenCount)
	estimatedCached := clampCachedPromptTokensInt(usage.promptTokens, int(math.Round(float64(reusablePrompt)*warmth)))
	return routerCacheAccounting{
		source:                "router_estimated",
		estimatedCachedTokens: estimatedCached,
		estimatedSavings:      estimatedCacheSavings(estimatedCached, pricing),
		confidence:            confidence,
	}
}

func estimatedCacheSavings(tokens int, pricing sessiontelemetry.TurnPricing) float64 {
	if tokens <= 0 || pricing.PromptPer1M <= 0 {
		return 0
	}
	cachedRate := pricing.CachedInputPer1M
	if cachedRate < 0 {
		cachedRate = 0
	}
	delta := pricing.PromptPer1M - cachedRate
	if delta <= 0 {
		return 0
	}
	return float64(tokens) * delta / 1_000_000.0
}

func clampFloat64(value, minValue, maxValue float64) float64 {
	return math.Max(minValue, math.Min(maxValue, value))
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func sessionTurnCost(usage responseUsageMetrics, pricing sessiontelemetry.TurnPricing) float64 {
	if pricing.PromptPer1M == 0 &&
		pricing.CompletionPer1M == 0 &&
		pricing.CachedInputPer1M == 0 &&
		pricing.Currency == "" {
		return 0
	}
	cached := clampCachedPromptTokensInt(usage.promptTokens, usage.cachedPromptTokens)
	uncachedPrompt := usage.promptTokens - cached
	return (float64(uncachedPrompt)*pricing.PromptPer1M +
		float64(cached)*pricing.CachedInputPer1M +
		float64(usage.completionTokens)*pricing.CompletionPer1M) / 1_000_000.0
}

// maybeEmitTransitionEvent records a ModelTransitionEvent on model change.
// Must be called after ctx.TTFTSeconds and ctx.CacheWarmthEstimate are set.
func maybeEmitTransitionEvent(ctx *RequestContext) {
	if ctx == nil || ctx.SessionID == "" || ctx.RequestModel == "" {
		return
	}
	if ctx.PreviousModel == "" || ctx.PreviousModel == ctx.RequestModel {
		return
	}

	previousResponseID := ""
	if ctx.ResponseAPICtx != nil {
		previousResponseID = ctx.ResponseAPICtx.PreviousResponseID
	}

	evt := sessiontelemetry.ModelTransitionEvent{
		SessionID:           ctx.SessionID,
		TurnIndex:           ctx.TurnIndex,
		FromModel:           ctx.PreviousModel,
		ToModel:             ctx.RequestModel,
		TTFTMs:              ctx.TTFTSeconds * 1000,
		CacheWarmthEstimate: ctx.CacheWarmthEstimate,
		PreviousResponseID:  previousResponseID,
		Timestamp:           time.Now(),
	}
	sessiontelemetry.RecordTransition(evt)
	metrics.RecordSessionModelTransition(evt.FromModel, evt.ToModel)
	metrics.RecordCacheWarmthEstimate(evt.ToModel, evt.CacheWarmthEstimate)
	logging.ComponentDebugEvent("session", "model_transition", map[string]interface{}{
		"session_id":            evt.SessionID,
		"turn_index":            evt.TurnIndex,
		"from_model":            evt.FromModel,
		"to_model":              evt.ToModel,
		"ttft_ms":               evt.TTFTMs,
		"cache_warmth_estimate": evt.CacheWarmthEstimate,
	})
}
