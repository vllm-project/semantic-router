package extproc

import (
	"context"
	"encoding/json"
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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

type responseUsageMetrics struct {
	promptTokens     int
	completionTokens int
}

func (r *OpenAIRouter) handleNonStreamingResponseBody(
	responseBody []byte,
	ctx *RequestContext,
	completionLatency time.Duration,
	initialBodyTransformed bool,
) *ext_proc.ProcessingResponse {
	usage := parseResponseUsage(responseBody, ctx.RequestModel)
	r.reportNonStreamingUsage(ctx, completionLatency, usage)
	r.updateResponseCache(ctx, responseBody)

	finalBody := r.translateResponseBodyForClient(ctx, responseBody)
	bodyMutation, headerMutation := buildInitialResponseMutations(
		finalBody,
		initialBodyTransformed || isResponseAPIRequest(ctx),
	)

	if jailbreakResponse := r.performResponseJailbreakDetection(ctx, responseBody); jailbreakResponse != nil {
		return jailbreakResponse
	}
	if hallucinationResponse := r.performHallucinationDetection(ctx, responseBody); hallucinationResponse != nil {
		return hallucinationResponse
	}

	r.scheduleResponseMemoryStore(ctx, responseBody)
	r.markUnverifiedFactualResponse(ctx)

	response := r.applyResponseWarnings(ctx, responseBody, bodyMutation, headerMutation)
	r.updateRouterReplayHallucinationStatus(ctx)
	r.attachRouterReplayResponse(ctx, finalBody, true)
	return response
}

func parseResponseUsage(responseBody []byte, model string) responseUsageMetrics {
	var parsed openai.ChatCompletion
	if err := json.Unmarshal(responseBody, &parsed); err != nil {
		logging.Errorf("Error parsing tokens from response: %v", err)
		metrics.RecordRequestError(model, "parse_error")
	}

	return responseUsageMetrics{
		promptTokens:     int(parsed.Usage.PromptTokens),
		completionTokens: int(parsed.Usage.CompletionTokens),
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
			InputTokens:  usage.promptTokens,
			OutputTokens: usage.completionTokens,
			TotalTokens:  totalTokens,
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

func (r *OpenAIRouter) updateResponseCache(ctx *RequestContext, responseBody []byte) {
	if ctx.RequestID == "" || responseBody == nil {
		return
	}

	// Skip cache store if a decision was selected but doesn't have semantic-cache enabled
	if ctx.VSRSelectedDecisionName != "" && r.Config != nil &&
		!r.Config.IsCacheEnabledForDecision(ctx.VSRSelectedDecisionName) {
		return
	}

	if ok, reason := ctx.HasPersonalizedContext(); ok {
		metrics.RecordCacheWriteSkipped(reason)
		logging.Infof("Skipping cache write for request ID %s: response has personalized context (reason=%s)", ctx.RequestID, reason)
		if span := trace.SpanFromContext(ctx.TraceContext); span.IsRecording() {
			span.SetAttributes(attribute.String(tracing.AttrCacheWriteSkippedReason, reason))
		}
		return
	}

	ttlSeconds := -1
	if r != nil && r.Config != nil {
		ttlSeconds = r.Config.GetCacheTTLSecondsForDecision(ctx.VSRSelectedDecisionName)
	}
	if err := r.Cache.UpdateWithResponse(ctx.RequestID, responseBody, ttlSeconds); err != nil {
		logging.Errorf("Error updating cache: %v", err)
		return
	}
	logging.Infof("Cache updated for request ID: %s", ctx.RequestID)
}

func (r *OpenAIRouter) translateResponseBodyForClient(
	ctx *RequestContext,
	responseBody []byte,
) []byte {
	if !isResponseAPIRequest(ctx) || r.ResponseAPIFilter == nil {
		return responseBody
	}

	translatedBody, err := r.ResponseAPIFilter.TranslateResponse(
		ctx.TraceContext,
		ctx.ResponseAPICtx,
		responseBody,
	)
	if err != nil {
		logging.Errorf("Response API translation error: %v", err)
		return responseBody
	}

	logging.Infof("Response API: Translated response to Response API format")
	return translatedBody
}

func buildInitialResponseMutations(
	finalBody []byte,
	bodyWasTransformed bool,
) (*ext_proc.BodyMutation, *ext_proc.HeaderMutation) {
	if !bodyWasTransformed {
		return nil, nil
	}

	return &ext_proc.BodyMutation{
			Mutation: &ext_proc.BodyMutation_Body{
				Body: finalBody,
			},
		}, &ext_proc.HeaderMutation{
			RemoveHeaders: []string{"content-length"},
		}
}

func (r *OpenAIRouter) scheduleResponseMemoryStore(ctx *RequestContext, responseBody []byte) {
	autoStoreEnabled := extractAutoStore(ctx)
	if !autoStoreEnabled && r.Config != nil && r.Config.Memory.AutoStore {
		logging.Infof("extractAutoStore: Falling back to global config, AutoStore=%v", r.Config.Memory.AutoStore)
		autoStoreEnabled = true
	}
	logging.Infof(
		"Memory store check: MemoryExtractor=%v, autoStore=%v, responseJailbreakPassed=%v",
		r.MemoryExtractor != nil,
		autoStoreEnabled,
		!ctx.ResponseJailbreakDetected,
	)
	if r.MemoryExtractor == nil || !autoStoreEnabled || ctx.ResponseJailbreakDetected {
		return
	}

	currentUserMessage := extractCurrentUserMessage(ctx)
	currentAssistantResponse := extractAssistantResponseText(responseBody)
	go func() {
		bgCtx := context.Background()
		sessionID, userID, history, err := extractMemoryInfo(ctx)
		if err != nil {
			logging.Errorf("Memory store failed: %v", err)
			return
		}

		logging.Infof(
			"Memory store: sessionID=%s, userID=%s, userMsg=%d chars, assistantMsg=%d chars, history=%d msgs",
			sessionID,
			userID,
			len(currentUserMessage),
			len(currentAssistantResponse),
			len(history),
		)

		if err := r.MemoryExtractor.ProcessResponseWithHistory(
			bgCtx,
			sessionID,
			userID,
			currentUserMessage,
			currentAssistantResponse,
			history,
		); err != nil {
			logging.Warnf("Memory store failed: %v", err)
		}
	}()
}

func (r *OpenAIRouter) markUnverifiedFactualResponse(ctx *RequestContext) {
	if ctx.VSRSelectedDecision == nil {
		return
	}

	hallucinationConfig := ctx.VSRSelectedDecision.GetHallucinationConfig()
	if hallucinationConfig != nil && hallucinationConfig.Enabled {
		r.checkUnverifiedFactualResponse(ctx)
	}
}

func (r *OpenAIRouter) applyResponseWarnings(
	ctx *RequestContext,
	responseBody []byte,
	bodyMutation *ext_proc.BodyMutation,
	headerMutation *ext_proc.HeaderMutation,
) *ext_proc.ProcessingResponse {
	response := buildResponseBodyContinueResponse(bodyMutation, headerMutation)
	modifiedBody := responseBody
	needsBodyMutation := false

	if ctx.ResponseJailbreakDetected {
		modifiedBody, response = r.applyResponseJailbreakWarning(response, ctx, modifiedBody)
	}
	if ctx.HallucinationDetected {
		modifiedBody, response = r.applyHallucinationWarning(response, ctx, modifiedBody)
		if string(modifiedBody) != string(responseBody) {
			needsBodyMutation = true
		}
	}
	if ctx.UnverifiedFactualResponse {
		modifiedBody, response = r.applyUnverifiedFactualWarning(response, ctx, modifiedBody)
		if string(modifiedBody) != string(responseBody) {
			needsBodyMutation = true
		}
	}
	if needsBodyMutation {
		bodyResponse := response.Response.(*ext_proc.ProcessingResponse_ResponseBody)
		bodyResponse.ResponseBody.Response.BodyMutation = &ext_proc.BodyMutation{
			Mutation: &ext_proc.BodyMutation_Body{
				Body: modifiedBody,
			},
		}
	}
	return response
}

func isResponseAPIRequest(ctx *RequestContext) bool {
	return ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.IsResponseAPIRequest
}
