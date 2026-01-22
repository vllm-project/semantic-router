package extproc

import (
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.opentelemetry.io/otel/attribute"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// logRoutingDecision logs routing decision with structured logging
func (r *OpenAIRouter) logRoutingDecision(ctx *RequestContext, reasonCode string, originalModel string, selectedModel string, decisionName string, reasoningEnabled bool, endpoint string) {
	effortForMetrics := ""
	if reasoningEnabled && decisionName != "" {
		effortForMetrics = r.getReasoningEffort(decisionName, selectedModel)
	}

	logging.LogEvent("routing_decision", map[string]interface{}{
		"reason_code":        reasonCode,
		"request_id":         ctx.RequestID,
		"original_model":     originalModel,
		"selected_model":     selectedModel,
		"decision":           decisionName,
		"reasoning_enabled":  reasoningEnabled,
		"reasoning_effort":   effortForMetrics,
		"routing_latency_ms": time.Since(ctx.ProcessingStartTime).Milliseconds(),
	})
	metrics.RecordRoutingReasonCode(reasonCode, selectedModel)
}

// recordRoutingDecision records routing decision with tracing
func (r *OpenAIRouter) recordRoutingDecision(ctx *RequestContext, decisionName string, originalModel string, matchedModel string, reasoningDecision entropy.ReasoningDecision) {
	// Start decision evaluation span
	routingCtx, routingSpan := tracing.StartDecisionSpan(ctx.TraceContext, decisionName)

	useReasoning := reasoningDecision.UseReasoning
	logging.Infof("Entropy-based reasoning decision for this query: %v on [%s] model (confidence: %.3f, reason: %s)",
		useReasoning, matchedModel, reasoningDecision.Confidence, reasoningDecision.DecisionReason)

	effortForMetrics := r.getReasoningEffort(decisionName, matchedModel)
	metrics.RecordReasoningDecision(decisionName, matchedModel, useReasoning, effortForMetrics)

	// Keep legacy attributes for backward compatibility
	tracing.SetSpanAttributes(routingSpan,
		attribute.String(tracing.AttrRoutingStrategy, "auto"),
		attribute.String(tracing.AttrRoutingReason, reasoningDecision.DecisionReason),
		attribute.String(tracing.AttrOriginalModel, originalModel),
		attribute.String(tracing.AttrSelectedModel, matchedModel),
		attribute.Bool(tracing.AttrReasoningEnabled, useReasoning),
		attribute.String(tracing.AttrReasoningEffort, effortForMetrics))

	// End decision span with evaluation results
	// matchedRules would come from signal evaluation, using empty slice for now
	tracing.EndDecisionSpan(routingSpan, float64(reasoningDecision.Confidence), []string{}, "auto")
	ctx.TraceContext = routingCtx
}

// trackVSRDecision tracks VSR decision information in context
// categoryName: the category from domain classification (MMLU category)
// decisionName: the decision name from DecisionEngine evaluation
func (r *OpenAIRouter) trackVSRDecision(ctx *RequestContext, categoryName string, decisionName string, matchedModel string, useReasoning bool) {
	ctx.VSRSelectedCategory = categoryName
	ctx.VSRSelectedDecisionName = decisionName
	ctx.VSRSelectedModel = matchedModel
	if useReasoning {
		ctx.VSRReasoningMode = "on"
	} else {
		ctx.VSRReasoningMode = "off"
	}
}

// setClearRouteCache sets the ClearRouteCache flag on the response
func (r *OpenAIRouter) setClearRouteCache(response *ext_proc.ProcessingResponse) {
	if response.GetRequestBody() != nil && response.GetRequestBody().GetResponse() != nil {
		response.GetRequestBody().GetResponse().ClearRouteCache = true
		logging.Debugf("Setting ClearRouteCache=true (feature enabled)")
	}
}

// recordRoutingLatency records the routing latency metric
func (r *OpenAIRouter) recordRoutingLatency(ctx *RequestContext) {
	routingLatency := time.Since(ctx.ProcessingStartTime)
	metrics.RecordModelRoutingLatency(routingLatency.Seconds())
}

// startRouterReplay begins capturing a replay record if the router_replay plugin is enabled
// for the matched decision. It is safe to call multiple times; only the first call is recorded.
func (r *OpenAIRouter) startRouterReplay(
	ctx *RequestContext,
	originalModel string,
	selectedModel string,
	decisionName string,
) {
	if ctx == nil || ctx.RouterReplayConfig == nil || !ctx.RouterReplayConfig.Enabled {
		return
	}
	if ctx.RouterReplayID != "" {
		return
	}

	// Get the recorder for this specific decision
	recorder := r.ReplayRecorders[decisionName]
	if recorder == nil {
		// Fall back to legacy single recorder if decision-specific one not found
		recorder = r.ReplayRecorder
		if recorder == nil {
			return
		}
	}

	cfg := ctx.RouterReplayConfig
	maxBodyBytes := cfg.MaxBodyBytes
	if maxBodyBytes <= 0 {
		maxBodyBytes = routerreplay.DefaultMaxBodyBytes
	}

	recorder.SetCapturePolicy(
		cfg.CaptureRequestBody,
		cfg.CaptureResponseBody,
		maxBodyBytes,
	)

	reasoningMode := ctx.VSRReasoningMode
	if reasoningMode == "" {
		reasoningMode = "off"
	}

	modelForRecord := selectedModel
	if modelForRecord == "" {
		modelForRecord = originalModel
	}

	rec := routerreplay.RoutingRecord{
		RequestID:     ctx.RequestID,
		Decision:      decisionName,
		Category:      ctx.VSRSelectedCategory,
		OriginalModel: originalModel,
		SelectedModel: modelForRecord,
		ReasoningMode: reasoningMode,
		Signals: routerreplay.Signal{
			Keyword:      ctx.VSRMatchedKeywords,
			Embedding:    ctx.VSRMatchedEmbeddings,
			Domain:       ctx.VSRMatchedDomains,
			FactCheck:    ctx.VSRMatchedFactCheck,
			UserFeedback: ctx.VSRMatchedUserFeedback,
			Preference:   ctx.VSRMatchedPreference,
		},
		Streaming: ctx.ExpectStreamingResponse,
		FromCache: ctx.VSRCacheHit,
	}

	// Attach request body directly; recorder will enforce capture + truncation
	if len(ctx.OriginalRequestBody) > 0 {
		rec.RequestBody = string(ctx.OriginalRequestBody)
	}

	replayID, err := recorder.AddRecord(rec)
	if err != nil {
		return
	}
	ctx.RouterReplayID = replayID
	ctx.RouterReplayRecorder = recorder

	if stored, ok := recorder.GetRecord(replayID); ok {
		logging.LogEvent(
			"router_replay_start",
			routerreplay.LogFields(stored, "router_replay_start"),
		)
	}
}

// updateRouterReplayStatus updates status metadata (status code, streaming/cache flags).
func (r *OpenAIRouter) updateRouterReplayStatus(ctx *RequestContext, status int, streaming bool) {
	if ctx == nil || ctx.RouterReplayID == "" {
		return
	}

	recorder := ctx.RouterReplayRecorder
	if recorder == nil {
		recorder = r.ReplayRecorder
	}
	if recorder == nil {
		return
	}

	err := recorder.UpdateStatus(ctx.RouterReplayID, status, ctx.VSRCacheHit, streaming)
	if err != nil {
		logging.Errorf("Failed to update router replay status: %v", err)
	}
}

// attachRouterReplayResponse stores response payload (if configured) and optionally logs completion.
func (r *OpenAIRouter) attachRouterReplayResponse(ctx *RequestContext, responseBody []byte, isFinal bool) {
	if ctx == nil || ctx.RouterReplayID == "" {
		return
	}

	recorder := ctx.RouterReplayRecorder
	if recorder == nil {
		recorder = r.ReplayRecorder
	}
	if recorder == nil {
		return
	}

	if len(responseBody) > 0 {
		_ = recorder.AttachResponse(ctx.RouterReplayID, responseBody)
	}

	if isFinal {
		if rec, ok := recorder.GetRecord(ctx.RouterReplayID); ok {
			logging.LogEvent(
				"router_replay_complete",
				routerreplay.LogFields(rec, "router_replay_complete"),
			)
		}
	}
}
