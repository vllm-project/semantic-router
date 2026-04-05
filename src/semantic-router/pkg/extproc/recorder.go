package extproc

import (
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.opentelemetry.io/otel/attribute"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// logRoutingDecision logs routing decision with structured logging
func (r *OpenAIRouter) logRoutingDecision(ctx *RequestContext, reasonCode string, originalModel string, selectedModel string, decisionName string, reasoningEnabled bool) {
	effortForMetrics := ""
	if reasoningEnabled && decisionName != "" {
		effortForMetrics = r.getReasoningEffort(decisionName, selectedModel)
	}

	logging.ComponentEvent("extproc", "routing_decision", map[string]interface{}{
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
	logging.ComponentDebugEvent("extproc", "reasoning_decision_applied", map[string]interface{}{
		"request_id":        ctx.RequestID,
		"decision":          decisionName,
		"original_model":    originalModel,
		"selected_model":    matchedModel,
		"reasoning_enabled": useReasoning,
		"confidence":        reasoningDecision.Confidence,
		"decision_reason":   reasoningDecision.DecisionReason,
	})

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
		logging.ComponentDebugEvent("extproc", "route_cache_clear_enabled", map[string]interface{}{
			"feature": "clear_route_cache",
		})
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
	if !shouldStartRouterReplay(ctx) {
		return
	}

	recorder := r.resolveReplayRecorder(decisionName)
	if recorder == nil {
		return
	}

	configureReplayRecorder(recorder, ctx.RouterReplayPluginConfig)
	record := buildReplayRoutingRecord(ctx, originalModel, selectedModel, decisionName)
	if !persistReplayRecord(ctx, recorder, record) {
		return
	}
}

func shouldStartRouterReplay(ctx *RequestContext) bool {
	if ctx == nil || ctx.RouterReplayPluginConfig == nil || !ctx.RouterReplayPluginConfig.Enabled {
		return false
	}
	return ctx.RouterReplayID == ""
}

func (r *OpenAIRouter) resolveReplayRecorder(decisionName string) *routerreplay.Recorder {
	recorder := r.ReplayRecorders[decisionName]
	if recorder != nil {
		return recorder
	}
	return r.ReplayRecorder
}

func configureReplayRecorder(
	recorder *routerreplay.Recorder,
	cfg *config.RouterReplayPluginConfig,
) {
	recorder.SetCapturePolicy(
		cfg.CaptureRequestBody,
		cfg.CaptureResponseBody,
		resolveReplayMaxBodyBytes(cfg.MaxBodyBytes),
	)
}

func buildReplayRoutingRecord(
	ctx *RequestContext,
	originalModel string,
	selectedModel string,
	decisionName string,
) routerreplay.RoutingRecord {
	guardrailsEnabled, jailbreakEnabled, piiEnabled, hallucinationEnabled := replayGuardrailState(ctx)
	decisionTier, decisionPriority := replayDecisionMetadata(ctx)
	record := routerreplay.RoutingRecord{
		RequestID:         ctx.RequestID,
		Decision:          decisionName,
		DecisionTier:      decisionTier,
		DecisionPriority:  decisionPriority,
		Category:          ctx.VSRSelectedCategory,
		OriginalModel:     originalModel,
		SelectedModel:     replaySelectedModel(originalModel, selectedModel),
		ReasoningMode:     replayReasoningMode(ctx),
		ConfidenceScore:   ctx.VSRSelectedDecisionConfidence,
		SelectionMethod:   ctx.VSRSelectionMethod,
		Signals:           replaySignalState(ctx),
		Projections:       replayProjectionState(ctx),
		ProjectionScores:  cloneReplayFloat64Map(ctx.VSRProjectionScores),
		SignalConfidences: cloneReplayFloat64Map(ctx.VSRSignalConfidences),
		SignalValues:      cloneReplayFloat64Map(ctx.VSRSignalValues),
		ToolTrace:         buildReplayRequestToolTrace(ctx),
		Streaming:         ctx.ExpectStreamingResponse,
		FromCache:         ctx.VSRCacheHit,

		GuardrailsEnabled: guardrailsEnabled,
		JailbreakEnabled:  jailbreakEnabled,
		PIIEnabled:        piiEnabled,

		JailbreakDetected:   ctx.JailbreakDetected,
		JailbreakType:       ctx.JailbreakType,
		JailbreakConfidence: ctx.JailbreakConfidence,

		ResponseJailbreakDetected:   ctx.ResponseJailbreakDetected,
		ResponseJailbreakType:       ctx.ResponseJailbreakType,
		ResponseJailbreakConfidence: ctx.ResponseJailbreakConfidence,

		PIIDetected: ctx.PIIDetected,
		PIIEntities: ctx.PIIEntities,
		PIIBlocked:  ctx.PIIBlocked,

		RAGEnabled:           ctx.RAGRetrievedContext != "",
		RAGBackend:           ctx.RAGBackend,
		RAGContextLength:     len(ctx.RAGRetrievedContext),
		RAGSimilarityScore:   ctx.RAGSimilarityScore,
		HallucinationEnabled: hallucinationEnabled,
	}
	if len(ctx.OriginalRequestBody) > 0 {
		record.RequestBody = string(ctx.OriginalRequestBody)
	}
	return record
}

func replayDecisionMetadata(ctx *RequestContext) (int, int) {
	if ctx == nil || ctx.VSRSelectedDecision == nil {
		return 0, 0
	}
	return ctx.VSRSelectedDecision.Tier, ctx.VSRSelectedDecision.Priority
}

func replayReasoningMode(ctx *RequestContext) string {
	if ctx.VSRReasoningMode == "" {
		return "off"
	}
	return ctx.VSRReasoningMode
}

func replaySelectedModel(originalModel string, selectedModel string) string {
	if selectedModel == "" {
		return originalModel
	}
	return selectedModel
}

func replaySignalState(ctx *RequestContext) routerreplay.Signal {
	return routerreplay.Signal{
		Keyword:      ctx.VSRMatchedKeywords,
		Embedding:    ctx.VSRMatchedEmbeddings,
		Domain:       ctx.VSRMatchedDomains,
		FactCheck:    ctx.VSRMatchedFactCheck,
		UserFeedback: ctx.VSRMatchedUserFeedback,
		Reask:        ctx.VSRMatchedReask,
		Preference:   ctx.VSRMatchedPreference,
		Language:     ctx.VSRMatchedLanguage,
		Context:      ctx.VSRMatchedContext,
		Structure:    ctx.VSRMatchedStructure,
		Complexity:   ctx.VSRMatchedComplexity,
		Modality:     ctx.VSRMatchedModality,
		Authz:        ctx.VSRMatchedAuthz,
		Jailbreak:    ctx.VSRMatchedJailbreak,
		PII:          ctx.VSRMatchedPII,
		KB:           ctx.VSRMatchedKB,
	}
}

func replayProjectionState(ctx *RequestContext) []string {
	if ctx == nil || len(ctx.VSRMatchedProjection) == 0 {
		return nil
	}
	return append([]string(nil), ctx.VSRMatchedProjection...)
}

func replayGuardrailState(ctx *RequestContext) (bool, bool, bool, bool) {
	if ctx.VSRSelectedDecision == nil {
		return false, false, false, false
	}
	jailbreakEnabled := ctx.VSRSelectedDecision.HasSignalType("jailbreak")
	piiEnabled := ctx.VSRSelectedDecision.HasSignalType("pii")
	hallucinationEnabled := false
	if hallucinationCfg := ctx.VSRSelectedDecision.GetHallucinationConfig(); hallucinationCfg != nil {
		hallucinationEnabled = hallucinationCfg.Enabled
	}
	return jailbreakEnabled || piiEnabled, jailbreakEnabled, piiEnabled, hallucinationEnabled
}

func persistReplayRecord(
	ctx *RequestContext,
	recorder *routerreplay.Recorder,
	record routerreplay.RoutingRecord,
) bool {
	replayID, err := recorder.AddRecord(record)
	if err != nil {
		return false
	}
	ctx.RouterReplayID = replayID
	ctx.RouterReplayRecorder = recorder

	if stored, ok := recorder.GetRecord(replayID); ok {
		logging.ComponentEvent(
			"extproc",
			"router_replay_start",
			routerreplay.LogFields(stored, "router_replay_start"),
		)
	}
	return true
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
		logging.ComponentErrorEvent("extproc", "router_replay_status_update_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"replay_id":  ctx.RouterReplayID,
			"error":      err.Error(),
		})
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
	if responseTrace := buildReplayResponseToolTrace(ctx, responseBody); responseTrace != nil {
		if stored, ok := recorder.GetRecord(ctx.RouterReplayID); ok {
			responseTrace = mergeReplayToolTraces(stored.ToolTrace, responseTrace)
		}
		if responseTrace != nil {
			_ = recorder.UpdateToolTrace(ctx.RouterReplayID, *responseTrace)
		}
	}

	if isFinal {
		if rec, ok := recorder.GetRecord(ctx.RouterReplayID); ok {
			logging.ComponentEvent(
				"extproc",
				"router_replay_complete",
				routerreplay.LogFields(rec, "router_replay_complete"),
			)
		}
	}
}

// updateRouterReplayHallucinationStatus updates the hallucination detection results in the replay record.
func (r *OpenAIRouter) updateRouterReplayHallucinationStatus(ctx *RequestContext) {
	if ctx == nil || ctx.RouterReplayID == "" {
		return
	}

	// Only update if hallucination detection was enabled
	if ctx.VSRSelectedDecision == nil {
		return
	}
	hallucinationConfig := ctx.VSRSelectedDecision.GetHallucinationConfig()
	if hallucinationConfig == nil || !hallucinationConfig.Enabled {
		return
	}

	recorder := ctx.RouterReplayRecorder
	if recorder == nil {
		recorder = r.ReplayRecorder
	}
	if recorder == nil {
		return
	}

	err := recorder.UpdateHallucinationStatus(
		ctx.RouterReplayID,
		ctx.HallucinationDetected,
		ctx.HallucinationConfidence,
		ctx.HallucinationSpans,
	)
	if err != nil {
		logging.ComponentErrorEvent("extproc", "router_replay_hallucination_update_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"replay_id":  ctx.RouterReplayID,
			"error":      err.Error(),
		})
	}
}

func (r *OpenAIRouter) updateRouterReplayUsageCost(ctx *RequestContext, usage routerreplay.UsageCost) {
	if ctx == nil || ctx.RouterReplayID == "" || usage.TotalTokens == nil {
		return
	}

	recorder := ctx.RouterReplayRecorder
	if recorder == nil {
		recorder = r.ReplayRecorder
	}
	if recorder == nil {
		return
	}

	if err := recorder.UpdateUsageCost(ctx.RouterReplayID, usage); err != nil {
		logging.ComponentErrorEvent("extproc", "router_replay_usage_update_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"replay_id":  ctx.RouterReplayID,
			"error":      err.Error(),
		})
	}
}
