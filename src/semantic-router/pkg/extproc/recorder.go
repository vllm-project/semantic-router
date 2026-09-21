package extproc

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strconv"
	"time"
	"unicode/utf8"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
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

	// Resolution is a point-in-time event; it does not pretend to measure backend execution.
	trace.SpanFromContext(ctx.TraceContext).AddEvent("routing.backend.resolved", trace.WithAttributes(
		attribute.String(tracing.AttrDecisionName, decisionName),
		attribute.String(tracing.AttrAlgorithm, ctx.VSRSelectionMethod),
		attribute.String(tracing.AttrRoutingReason, reasoningDecision.DecisionReason),
		attribute.String(tracing.AttrOriginalModel, originalModel),
		attribute.String(tracing.AttrSelectedModel, matchedModel),
		attribute.Bool(tracing.AttrReasoningEnabled, useReasoning),
		attribute.String(tracing.AttrReasoningEffort, effortForMetrics)))
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

// recordRoutingLatency records the routing latency metric
func (r *OpenAIRouter) recordRoutingLatency(ctx *RequestContext) {
	routingLatency := time.Since(ctx.ProcessingStartTime)
	ctx.RoutingLatency = routingLatency
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
	if !shouldStartRouterReplay(ctx) || !r.replayAllowedForRequest(ctx) {
		return
	}

	populateReplaySessionIfNeeded(ctx)

	recorder := r.resolveReplayRecorder(ctx, decisionName)
	if recorder == nil {
		return
	}

	configureReplayRecorder(recorder, ctx.RouterReplayPluginConfig)
	record := buildReplayRoutingRecord(ctx, originalModel, selectedModel, decisionName)
	r.populateReplayIdentity(&record, ctx)
	if !persistReplayRecord(ctx, recorder, record) {
		return
	}
}

// populateReplayIdentity records the same explicit identity used by protection,
// including custom header names, without enabling protection or changing request
// state. Optional Responses lineage remains the fallback for older clients.
func (r *OpenAIRouter) populateReplayIdentity(record *routerreplay.RoutingRecord, ctx *RequestContext) {
	if r == nil || ctx == nil || record == nil {
		return
	}
	cfg := config.RouterLearningProtectionConfig{}
	if r.Config != nil {
		cfg = r.Config.RouterLearning.Protection
	}
	identity, ok := r.protectionIdentity(ctx, cfg)
	if !ok {
		return
	}
	record.SessionID = identity.sessionID
	if identity.conversationID != "" {
		record.ConversationID = identity.conversationID
	}
	// Persist the key the gate resolved, so outcome ingest can land feedback
	// in the same window even when the session id alone keys it wider.
	if record.Learning != nil {
		record.Learning.ProtectionStateKey = routingLearningStateKey(ctx)
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

// replayUserTurnIndex groups tool continuations with the user message that
// started them. Protocol normalization represents tool results as RoleTool,
// including Anthropic tool_result blocks and retained Responses history.
// Prefer the original snapshot so context trimming or Memory cannot renumber
// turns. This history contains only messages actually available to the router.
// Keep the invocation index as a fallback when no user history is visible.
func replayUserTurnIndex(ctx *RequestContext) int {
	var messages []llmprotocol.Message
	if ctx.OriginalContextHistory != nil {
		messages = ctx.OriginalContextHistory.Conversation().Messages
	} else if ctx.SemanticRequest != nil {
		messages = ctx.SemanticRequest.Messages
	}
	userMessages := 0
	for _, message := range messages {
		if message.Role == llmprotocol.RoleUser {
			userMessages++
		}
	}
	if userMessages > 0 {
		return userMessages - 1
	}
	return ctx.TurnIndex
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
		RequestID:                ctx.RequestID,
		SessionID:                ctx.SessionID,
		TurnIndex:                replayUserTurnIndex(ctx),
		Decision:                 decisionName,
		Recipe:                   string(ctx.Routing.RecipeName()),
		DecisionTier:             decisionTier,
		DecisionPriority:         decisionPriority,
		Category:                 ctx.VSRSelectedCategory,
		OriginalModel:            originalModel,
		SelectedModel:            replaySelectedModel(selectedModel),
		ReasoningMode:            replayReasoningMode(ctx),
		ConfidenceScore:          ctx.VSRSelectedDecisionConfidence,
		ConfidenceScoreAvailable: ctx.VSRSelectedDecisionConfidenceScored,
		SelectionMethod:          ctx.VSRSelectionMethod,
		RouteDiagnostics:         buildReplayRouteDiagnostics(ctx, originalModel, selectedModel, decisionName, decisionTier, decisionPriority),
		Learning:                 buildReplayLearningDiagnostics(ctx),
		SessionPolicy:            sessionPolicyMapForReplay(ctx),
		Signals:                  replaySignalState(ctx),
		Projections:              replayProjectionState(ctx),
		ProjectionScores:         cloneReplayFloat64Map(ctx.VSRProjectionScores),
		ProjectionTrace:          cloneProjectionTraceForReplay(ctx.VSRProjectionTrace),
		SignalConfidences:        cloneReplayFloat64Map(ctx.VSRSignalConfidences),
		SignalErrorMatches:       cloneReplayBoolMap(ctx.VSRSignalErrorMatches),
		SignalValues:             cloneReplayFloat64Map(ctx.VSRSignalValues),
		ToolTrace:                buildReplayRequestToolTrace(ctx),
		Streaming:                ctx.ExpectStreamingResponse,
		FromCache:                ctx.VSRCacheHit,

		GuardrailsEnabled: guardrailsEnabled,
		JailbreakEnabled:  jailbreakEnabled,
		PIIEnabled:        piiEnabled,

		JailbreakDetected:       ctx.JailbreakDetected,
		JailbreakType:           ctx.JailbreakType,
		JailbreakConfidence:     ctx.JailbreakConfidence,
		JailbreakScoreAvailable: ctx.JailbreakScoreAvailable,
		JailbreakDecision:       ctx.JailbreakDecision,

		ResponseJailbreakDetected:       ctx.ResponseJailbreakDetected,
		ResponseJailbreakType:           ctx.ResponseJailbreakType,
		ResponseJailbreakConfidence:     ctx.ResponseJailbreakConfidence,
		ResponseJailbreakScoreAvailable: ctx.ResponseJailbreakScoreAvailable,
		ResponseJailbreakDecision:       ctx.ResponseJailbreakDecision,

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
		replayRequest := replaySafeSemanticRequest(*ctx.SemanticRequest)
		if requestBody, err := cache.MarshalSemanticRequest(replayRequest); err == nil {
			record.RequestBody = string(requestBody)
		}
		record.Prompt, record.ToolDefinitions = extractSemanticPromptAndTools(&replayRequest)
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
		Safety:        ctx.VSRMatchedSafety,
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
	if ctx == nil || ctx.RouterReplayID == "" || !r.replayAllowedForRequest(ctx) {
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
		attachPrimaryOutputDigest(ctx, recorder)
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
			ScoreAvailable:          span.ScoreAvailable,
			NLILabel:                span.NLILabel,
			NLIConfidence:           span.NLIConfidence,
			NLIScoreAvailable:       span.NLIScoreAvailable,
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
		routerreplay.HallucinationScore{Available: ctx.HallucinationScoreAvailable, Kind: ctx.HallucinationScoreKind},
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

// responseStageStreamingNotEnforced is the enforcement a streamed response
// gets: none. Its answer exists as a whole for the first time when the bytes
// are already with the client, so no plugin can act on the observation.
const responseStageStreamingNotEnforced = "not_enforced_streaming"

// recordResponseStageEnforcement records what acted on a response-stage
// observation. A buffered response names the action the selected decision's
// plugin applies, or nothing when the decision carries no enabled plugin. A
// streamed response names no action at all, because none ran: saying so is the
// point, since an "action" the record names but nothing applied would read as
// enforcement that happened.
func recordResponseStageEnforcement(outcome *routerreplay.Outcome, ctx *RequestContext, action string) {
	if ctx.IsStreamingResponse {
		outcome.Metadata["enforcement"] = responseStageStreamingNotEnforced
		return
	}
	if action != "" {
		outcome.Metadata["action"] = action
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
	recordResponseStageEnforcement(&outcome, ctx, action)
	if code, failed := ctx.VSRSignalErrors[key]; failed {
		outcome.Verdict = "unavailable"
		outcome.Reason = code
		outcome.Metadata["score_available"] = "false"
		if ctx.ResponseJailbreakType == classification.JailbreakClassificationErrorType {
			outcome.Metadata["policy_match"] = "true"
		}
		return outcome
	}
	if value, available := ctx.VSRSignalConfidences[key]; available {
		outcome.Score = value
		outcome.Metadata["score_available"] = "true"
	} else {
		outcome.Metadata["score_available"] = "false"
	}
	if decision := ctx.VSRResponseJailbreakDecision; decision != nil {
		outcome.Metadata["source_label"] = decision.SourceLabel
		outcome.Metadata["label"] = decision.Label
	}
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

// recordRouterReplayHallucination appends the response-stage hallucination
// observation to the replay record, one outcome per hallucination rule, the
// way recordRouterReplayResponseJailbreak does for jailbreak rules. A rule the
// request never evaluated (the fact-check signal said the prompt makes no
// claims worth grounding) is recorded as not applicable, so the record says
// why nothing was checked rather than saying nothing.
func (r *OpenAIRouter) recordRouterReplayHallucination(ctx *RequestContext) {
	if ctx == nil || ctx.RouterReplayID == "" {
		return
	}
	rules := r.hallucinationRules(ctx)
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
	action := ""
	if r.isHallucinationEnabledForDecision(ctx.VSRSelectedDecision) {
		action = r.getHallucinationActionForDecision(ctx.VSRSelectedDecision)
	}
	for _, rule := range rules {
		outcome := hallucinationReplayOutcome(ctx, rule, now, action)
		if err := recorder.AppendOutcome(ctx.RouterReplayID, outcome); err != nil {
			logging.ComponentErrorEvent("extproc", "router_replay_hallucination_outcome_failed", map[string]interface{}{
				"request_id": ctx.RequestID,
				"replay_id":  ctx.RouterReplayID,
				"rule":       rule.Name,
				"error":      err.Error(),
			})
		}
	}
}

func hallucinationReplayOutcome(ctx *RequestContext, rule config.HallucinationRule, now time.Time, action string) routerreplay.Outcome {
	key := signalKey(config.SignalTypeHallucination, rule.Name)
	outcome := routerreplay.Outcome{
		Timestamp: now,
		Source:    "router",
		Target:    key,
		Verdict:   "not_applicable",
		Reason:    "fact_check_not_needed",
		Metadata: map[string]string{
			"signal":    config.SignalTypeHallucination,
			"direction": config.SignalDirectionResponse,
			"use_nli":   strconv.FormatBool(rule.UseNLI),
		},
	}
	if ctx.VSRSelectedDecisionName != "" {
		outcome.Metadata["decision"] = ctx.VSRSelectedDecisionName
	}
	recordResponseStageEnforcement(&outcome, ctx, action)
	if code, failed := ctx.VSRSignalErrors[key]; failed {
		outcome.Verdict = "unavailable"
		outcome.Reason = code
		return outcome
	}
	score, observed := ctx.VSRSignalConfidences[key]
	if !observed && ctx.VSRHallucinationEvidence == nil {
		return outcome
	}
	outcome.Verdict = "not_detected"
	outcome.Reason = ""
	outcome.Score = score
	if evidence := ctx.VSRHallucinationEvidence; evidence != nil {
		outcome.Metadata["spans"] = strconv.Itoa(len(evidence.Spans))
		outcome.Metadata["score_available"] = strconv.FormatBool(evidence.ScoreAvailable)
		if evidence.ScoreAvailable {
			outcome.Score = float64(evidence.Confidence)
			outcome.Metadata["score_kind"] = evidence.ScoreKind
		}
	}
	for _, matched := range ctx.VSRMatchedHallucination {
		if matched == rule.Name {
			outcome.Verdict = "detected"
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

// primaryResponseOutcomeSource marks the outcome that carries the digest of
// what the selected model answered. Shadow dispatch writes its arms the same
// way, so an offline comparison reads both sides through one path instead of
// hashing a stored body on one side and a decoded answer on the other.
const primaryResponseOutcomeSource = "primary_response"

// recordPrimaryOutputDigest hashes what the selected model answered, using the
// same contract a shadow arm is hashed under: the assistant text of the decoded
// response, never the encoded protocol body, which carries a JSON envelope, a
// response id and a usage block that a shadow digest never sees.
//
// It runs before any response-stage plugin, because a body warning prepends
// router text to the same response in place. Hashing after that would credit
// the warning to the model and the two arms would stop comparing.
func recordPrimaryOutputDigest(ctx *RequestContext, response *llmprotocol.Response) {
	if ctx == nil || response == nil || ctx.PrimaryOutputDigest != "" {
		return
	}
	text := semanticResponseText(*response)
	if text == "" {
		return
	}
	sum := sha256.Sum256([]byte(text))
	ctx.PrimaryOutputDigest = hex.EncodeToString(sum[:])
	ctx.PrimaryOutputChars = utf8.RuneCountInString(text)
}

// attachPrimaryOutputDigest persists the digest captured before the response
// was rewritten, so an offline comparison reads both arms through one contract.
func attachPrimaryOutputDigest(ctx *RequestContext, recorder *routerreplay.Recorder) {
	if ctx.PrimaryOutputDigest == "" {
		return
	}
	outcome := routerreplay.Outcome{
		Timestamp: time.Now().UTC(),
		Source:    primaryResponseOutcomeSource,
		Target:    "model",
		TargetRef: ctx.VSRSelectedModel,
		Verdict:   "completed",
		Metadata: map[string]string{
			"response_sha256": ctx.PrimaryOutputDigest,
			"response_chars":  strconv.Itoa(ctx.PrimaryOutputChars),
		},
	}
	if err := recorder.AppendOutcome(ctx.RouterReplayID, outcome); err != nil {
		logging.ComponentErrorEvent("extproc", "primary_output_digest_persist_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"replay_id":  ctx.RouterReplayID,
			"error":      err.Error(),
		})
	}
}
