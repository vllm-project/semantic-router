package extproc

import (
	"encoding/json"
	"strconv"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.opentelemetry.io/otel/attribute"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
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
		effortForMetrics = r.getReasoningEffort(ctx.VSRSelectedDecision, selectedModel)
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

	effortForMetrics := r.getReasoningEffort(ctx.VSRSelectedDecision, matchedModel)
	metrics.RecordReasoningDecision(requestDecisionStateKey(ctx), matchedModel, useReasoning, effortForMetrics)

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

	populateReplaySessionIfNeeded(ctx)

	recorder := r.resolveReplayRecorder(ctx, decisionName)
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

// populateReplaySessionIfNeeded derives session fields from neutral state when
// replay starts before the regular request-preparation phase.
func populateReplaySessionIfNeeded(ctx *RequestContext) {
	if ctx == nil || ctx.SemanticRequest == nil {
		return
	}
	populateSessionTransitionFields(ctx)
}

func (r *OpenAIRouter) resolveReplayRecorder(ctx *RequestContext, decisionName string) *routerreplay.Recorder {
	recipeName := config.DefaultRecipeName
	if ctx != nil && ctx.Routing.RecipeName() != "" {
		recipeName = ctx.Routing.RecipeName()
	}
	recorder := r.ReplayRecorders[config.RoutingDecisionKey(recipeName, decisionName)]
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
	recorder.SetMaxToolTraceBytes(cfg.MaxToolTraceBytes)
	recorder.SetMaxToolTraceSteps(cfg.MaxToolTraceSteps)
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
		SessionID:         ctx.SessionID,
		TurnIndex:         ctx.TurnIndex,
		Decision:          decisionName,
		Recipe:            string(ctx.Routing.RecipeName()),
		DecisionTier:      decisionTier,
		DecisionPriority:  decisionPriority,
		Category:          ctx.VSRSelectedCategory,
		OriginalModel:     originalModel,
		SelectedModel:     replaySelectedModel(selectedModel),
		ReasoningMode:     replayReasoningMode(ctx),
		ConfidenceScore:   ctx.VSRSelectedDecisionConfidence,
		SelectionMethod:   ctx.VSRSelectionMethod,
		RouteDiagnostics:  buildReplayRouteDiagnostics(ctx, originalModel, selectedModel, decisionName, decisionTier, decisionPriority),
		Learning:          buildReplayLearningDiagnostics(ctx),
		SessionPolicy:     sessionPolicyMapForReplay(ctx),
		Signals:           replaySignalState(ctx),
		Projections:       replayProjectionState(ctx),
		ProjectionScores:  cloneReplayFloat64Map(ctx.VSRProjectionScores),
		ProjectionTrace:   cloneProjectionTraceForReplay(ctx.VSRProjectionTrace),
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
		CacheSimilarity:      ctx.VSRCacheSimilarity,
		CacheHitKind:         ctx.VSRCacheHitKind,
		CacheSource:          ctx.VSRCacheSource,
		CacheEntryAgeSeconds: ctx.VSRCacheEntryAgeSeconds,
		CacheTTLSeconds:      ctx.VSRCacheTTLSeconds,
		ContextTokenCount:    ctx.VSRContextTokenCount,
		HallucinationEnabled: hallucinationEnabled,
	}
	if state := ctx.ResponseObjectState; state != nil {
		record.PreviousResponseID = state.PreviousResponseID
		record.ConversationID = state.ConversationID
	}
	if ctx.SemanticRequest != nil {
		if requestBody, err := cache.MarshalSemanticRequest(*ctx.SemanticRequest); err == nil {
			record.RequestBody = string(requestBody)
		}
	}

	// Extract structured fields from neutral IR before recorder truncation.
	record.Prompt, record.ToolDefinitions = extractSemanticPromptAndTools(ctx.SemanticRequest)

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

func replaySelectedModel(selectedModel string) string {
	return selectedModel
}

func replaySignalState(ctx *RequestContext) routerreplay.Signal {
	return routerreplay.Signal{
		Keyword:       ctx.VSRMatchedKeywords,
		Embedding:     ctx.VSRMatchedEmbeddings,
		Domain:        ctx.VSRMatchedDomains,
		FactCheck:     ctx.VSRMatchedFactCheck,
		UserFeedback:  ctx.VSRMatchedUserFeedback,
		Reask:         ctx.VSRMatchedReask,
		Preference:    ctx.VSRMatchedPreference,
		Language:      ctx.VSRMatchedLanguage,
		Context:       ctx.VSRMatchedContext,
		Structure:     ctx.VSRMatchedStructure,
		Complexity:    ctx.VSRMatchedComplexity,
		Modality:      ctx.VSRMatchedModality,
		Authz:         ctx.VSRMatchedAuthz,
		Jailbreak:     ctx.VSRMatchedJailbreak,
		PII:           ctx.VSRMatchedPII,
		KB:            ctx.VSRMatchedKB,
		Conversation:  ctx.VSRMatchedConversation,
		Event:         ctx.VSRMatchedEvent,
		Metadata:      ctx.VSRMatchedMetadata,
		Classifier:    ctx.VSRMatchedClassifier,
		InputModality: ctx.VSRMatchedInputModality,
	}
}

func replayProjectionState(ctx *RequestContext) []string {
	if ctx == nil || len(ctx.VSRMatchedProjection) == 0 {
		return nil
	}
	return append([]string(nil), ctx.VSRMatchedProjection...)
}

func cloneReplayInterfaceMap(values map[string]interface{}) map[string]interface{} {
	if values == nil {
		return nil
	}
	b, err := json.Marshal(values)
	if err != nil {
		return nil
	}
	var cloned map[string]interface{}
	if err := json.Unmarshal(b, &cloned); err != nil {
		return nil
	}
	return cloned
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
		logging.ComponentErrorEvent("extproc", "router_replay_persist_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"decision":   record.Decision,
			"error":      err.Error(),
		})
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

func (r *OpenAIRouter) finalizeRouterReplay(
	ctx *RequestContext,
	state string,
	reason string,
) {
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

	if err := recorder.FinalizeLifecycle(ctx.RouterReplayID, state, reason); err != nil {
		logging.ComponentErrorEvent("extproc", "router_replay_lifecycle_update_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"replay_id":  ctx.RouterReplayID,
			"state":      state,
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
		state := routerreplay.LifecycleCompleted
		reason := "response_complete"
		if ctx.UpstreamStatusCode >= 400 {
			state = routerreplay.LifecycleFailed
			reason = "upstream_error_response"
		}
		r.finalizeRouterReplay(ctx, state, reason)
		if rec, ok := recorder.GetRecord(ctx.RouterReplayID); ok {
			logging.ComponentEvent(
				"extproc",
				"router_replay_complete",
				routerreplay.LogFields(rec, "router_replay_complete"),
			)
		}
	}
}

// hallucinationSpanDetailsForReplay converts NLI span analysis into the
// replay store's shape. Returns nil when NLI detection did not run for this
// request, so basic (non-NLI) detection continues to persist plain spans only.
func hallucinationSpanDetailsForReplay(info *EnhancedHallucinationInfo) []routerreplay.HallucinationSpan {
	if info == nil {
		return nil
	}
	details := make([]routerreplay.HallucinationSpan, len(info.Spans))
	for i, span := range info.Spans {
		details[i] = routerreplay.HallucinationSpan{
			Text:                    span.Text,
			Start:                   span.Start,
			End:                     span.End,
			HallucinationConfidence: span.HallucinationConfidence,
			NLILabel:                span.NLILabel,
			NLIConfidence:           span.NLIConfidence,
			Severity:                span.Severity,
			Explanation:             span.Explanation,
		}
	}
	return details
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
		hallucinationSpanDetailsForReplay(ctx.EnhancedHallucinationInfo),
	)
	if err != nil {
		logging.ComponentErrorEvent("extproc", "router_replay_hallucination_update_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"replay_id":  ctx.RouterReplayID,
			"error":      err.Error(),
		})
	}
}

// recordRouterReplayResponseJailbreak appends the response-stage jailbreak
// observation to the replay record, one outcome per response-direction rule.
// The record is written while the request is routed, before the model has
// answered, so the request-stage signal maps it carries cannot hold this;
// outcomes are the append-only post-route channel every store implements.
// Each outcome names the rule under the signal key a request-direction rule
// uses, its verdict (detected, not_detected, unavailable), the score it
// thresholded or the failure code when it could not resolve, and the action
// the selected decision's plugin applied.
func (r *OpenAIRouter) recordRouterReplayResponseJailbreak(ctx *RequestContext) {
	if ctx == nil || ctx.RouterReplayID == "" {
		return
	}
	rules := r.responseJailbreakRules(ctx)
	if len(rules) == 0 {
		return
	}
	recorder := ctx.RouterReplayRecorder
	if recorder == nil {
		recorder = r.ReplayRecorder
	}
	if recorder == nil {
		return
	}
	now := time.Now().UTC()
	action := r.responseJailbreakPluginAction(ctx)
	for _, rule := range rules {
		outcome := responseJailbreakReplayOutcome(ctx, rule, now, action)
		if err := recorder.AppendOutcome(ctx.RouterReplayID, outcome); err != nil {
			logging.ComponentErrorEvent("extproc", "router_replay_response_jailbreak_outcome_failed", map[string]interface{}{
				"request_id": ctx.RequestID,
				"replay_id":  ctx.RouterReplayID,
				"rule":       rule.Name,
				"error":      err.Error(),
			})
		}
	}
}

func responseJailbreakReplayOutcome(ctx *RequestContext, rule config.JailbreakRule, now time.Time, action string) routerreplay.Outcome {
	key := signalKey(config.SignalTypeJailbreak, rule.Name)
	outcome := routerreplay.Outcome{
		Timestamp: now,
		Source:    "router",
		Target:    key,
		Verdict:   "not_detected",
		Metadata: map[string]string{
			"signal":    config.SignalTypeJailbreak,
			"direction": config.SignalDirectionResponse,
			"threshold": strconv.FormatFloat(float64(rule.Threshold), 'f', -1, 32),
		},
	}
	if ctx.VSRSelectedDecisionName != "" {
		outcome.Metadata["decision"] = ctx.VSRSelectedDecisionName
	}
	if action != "" {
		outcome.Metadata["action"] = action
	}
	if code, failed := ctx.VSRSignalErrors[key]; failed {
		outcome.Verdict = "unavailable"
		outcome.Reason = code
		return outcome
	}
	outcome.Score = ctx.VSRSignalConfidences[key]
	for _, matched := range ctx.VSRMatchedResponseJailbreak {
		if matched == rule.Name {
			outcome.Verdict = "detected"
			if ctx.VSRResponseJailbreakType != "" {
				outcome.Metadata["type"] = ctx.VSRResponseJailbreakType
			}
			break
		}
	}
	return outcome
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
