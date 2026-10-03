package extproc

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/historyreset"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// bindHistoryResetPolicy resolves the selected decision's reset policy before
// any early response or cache lookup. An enabled policy bypasses the response
// cache in both directions: the cache identity covers static configuration but
// not the per-request topic result, recovery availability, or the receipt that
// an enabled policy must produce, so a hit could otherwise skip an authorized
// evaluation.
func bindHistoryResetPolicy(ctx *RequestContext) {
	if ctx == nil || ctx.VSRSelectedDecision == nil {
		return
	}
	policy := ctx.VSRSelectedDecision.GetHistoryResetConfig()
	if !policy.IsEnabled() {
		return
	}
	ctx.HistoryResetPolicy = policy
	ctx.CacheReadBypass = true
	ctx.CacheWriteBypass = true
}

// prepareContextHistorySteps registers the configured history actions on the
// request. It runs after RAG and Memory enrichment and immediately before the
// shared context stage, so the policy sees the same messages the executor will
// transform. It must not construct the request IR; the shared stage owns that.
func (r *OpenAIRouter) prepareContextHistorySteps(ctx *RequestContext, request *llmprotocol.Request) {
	if ctx == nil || !ctx.HistoryResetPolicy.IsEnabled() || ctx.HistoryResetAction != nil {
		return
	}
	blocked := historyResetBlockedReason(ctx, request)
	recoverySettings, settingsErr := contextRecoverySettingsForRequest(ctx)
	if settingsErr != nil {
		blocked = historyreset.ReasonRecoveryUnavailable
	}
	if blocked == "" && historyResetEvidenceBinding(ctx) == "" {
		// Without a resolved original history there is nothing to bind evidence
		// to, so no result can be shown to describe this request.
		blocked = historyreset.ReasonHistoryUnresolved
	}
	var writer historyreset.RecoveryWriter
	var detached map[int]llmprotocol.Message
	if blocked == "" {
		writer, detached, blocked = r.historyResetRecovery(ctx, request)
	}
	policy := historyResetPolicy(ctx, ctx.HistoryResetPolicy, recoverySettings)
	// A request that is already terminal cannot be changed by any result, so
	// the producer is not asked: evaluating a topic signal can be expensive,
	// and its answer could only be discarded.
	var trigger historyreset.TriggerResult
	if blocked == "" {
		trigger = r.historyResetTrigger(ctx, policy)
	}
	action := historyreset.NewAction(policy, trigger, blocked)
	if writer != nil {
		action = action.WithRecovery(writer, detached)
	}
	ctx.HistoryResetAction = action
	ctx.ContextHistorySteps = append(ctx.ContextHistorySteps, action.Step())
}

// prepareLooperContextHistorySteps registers the history actions for an
// internal Looper hop. The hop continues a public turn the parent request
// already evaluated, so it inherits that completion instead of treating
// appended tool results or a model change as a new topic-change event.
func prepareLooperContextHistorySteps(ctx *RequestContext) {
	if ctx == nil || !ctx.HistoryResetPolicy.IsEnabled() || ctx.HistoryResetAction != nil {
		return
	}
	action := historyreset.NewInheritedAction(historyResetPolicy(ctx, ctx.HistoryResetPolicy, nil))
	ctx.HistoryResetAction = action
	ctx.ContextHistorySteps = append(ctx.ContextHistorySteps, action.Step())
}

// historyResetPolicy translates validated configuration into the policy the
// pure action executes with. Configuration cannot widen shared eligibility.
func historyResetPolicy(
	ctx *RequestContext,
	configured *config.HistoryResetPluginConfig,
	recovery *config.ContextCompressionRecoveryConfig,
) historyreset.Policy {
	limits := configured.EffectiveLimits()
	confidence, _ := configured.EffectiveMinConfidence()
	policy := historyreset.Policy{
		MinConfidence:   confidence,
		MaxHistoryTurns: limits.MaxHistoryTurns,
		MaxHistoryBytes: limits.MaxHistoryBytes,
		FailClosed:      configured.EffectiveFailureMode() == config.HistoryResetFailureClosed,
		Binding:         historyResetEvidenceBinding(ctx),
		Timeout:         time.Duration(limits.TimeoutMs) * time.Millisecond,
	}
	if configured.Trigger != nil {
		policy.Signal = configured.Trigger.Signal
		policy.AcceptedVersions = append([]string(nil), configured.Trigger.AcceptedVersions...)
	}
	// The recovery budget comes from the merged request-level contract, not
	// from this plugin's own value: one action must not be able to persist
	// more than the contract another action agreed to.
	policy.MaxRecoveryBytes = effectiveRecoveryBytes(recovery)
	return policy
}

// historyResetTrigger resolves the topic-continuity result for this request.
// The producer is asked here, after the permitted history has been resolved
// and the request's binding computed, so a signal evaluated earlier in the
// pipeline cannot classify a different view of the conversation than the one
// the action transforms.
//
// A result already present on the request context wins: internal follow-ups
// and tests supply one directly. When nothing produces a result the action
// treats the evidence as missing and applies its failure mode rather than
// assuming continuation or change.
func (r *OpenAIRouter) historyResetTrigger(
	ctx *RequestContext,
	policy historyreset.Policy,
) historyreset.TriggerResult {
	if ctx.HistoryResetTrigger != nil {
		return *ctx.HistoryResetTrigger
	}
	if r == nil || r.HistoryResetTriggers == nil {
		return historyreset.TriggerResult{}
	}
	callContext := ctx.TraceContext
	if callContext == nil {
		callContext = context.Background()
	}
	if ctx.OriginalContextHistory == nil {
		// Nothing resolved the history, so there is no conversation to
		// classify and no binding to verify a result against.
		return historyreset.TriggerResult{}
	}
	result, ok := r.HistoryResetTriggers.TopicContinuity(callContext, historyreset.TriggerRequest{
		Signal:  policy.Signal,
		Binding: policy.Binding,
		History: ctx.OriginalContextHistory.Conversation(),
	})
	if !ok {
		return historyreset.TriggerResult{}
	}
	ctx.HistoryResetTrigger = &result
	return result
}

// effectiveRecoveryBytes reports the per-request payload bound every context
// action shares. An omitted bound falls back to a documented default so a
// configuration that enables recovery can never persist without one.
func effectiveRecoveryBytes(recovery *config.ContextCompressionRecoveryConfig) int {
	if recovery == nil {
		return 0
	}
	if recovery.MaxBytesPerRequest > 0 {
		return recovery.MaxBytesPerRequest
	}
	return defaultContextRecoveryBytesPerRequest
}

// historyResetBlockedReason reports a terminal condition established before
// planning. Reset is semantic-only; recoverability is resolved separately
// because it depends on runtime stores rather than the request shape.
func historyResetBlockedReason(ctx *RequestContext, request *llmprotocol.Request) string {
	if request == nil || ctx.SemanticRequest == nil {
		return historyreset.ReasonUnsupportedRepresentation
	}
	return ctx.HistoryResetBlocked
}

// finalizeHistoryResetDiagnostics reconciles the action's bounded diagnostic
// with the executor's committed receipt. It runs on both the success and the
// failure path so a rejected plan still reports why, and it never records more
// removals than the executor committed.
func finalizeHistoryResetDiagnostics(ctx *RequestContext, ir *contextcompression.RequestIR) {
	if ctx == nil || ctx.HistoryResetAction == nil || ir == nil {
		return
	}
	diagnostics := ctx.HistoryResetAction.Reconcile(ir.Transformations.Receipts())
	ctx.HistoryResetDiagnostics = &diagnostics
	metrics.RecordHistoryResetEvaluation(
		ctx.VSRSelectedDecisionName,
		string(diagnostics.Outcome),
		diagnostics.Reason,
		diagnostics.RemovedTurns,
		diagnostics.RemovedMessages,
	)
	metrics.RecordHistoryResetRecovery(diagnostics.RecoveryStatus)
	logging.ComponentEvent("extproc", "history_reset_evaluated", map[string]interface{}{
		"request_id":         ctx.RequestID,
		"signal":             diagnostics.Signal,
		"scope":              diagnostics.Scope,
		"trigger_class":      string(diagnostics.TriggerClass),
		"outcome":            string(diagnostics.Outcome),
		"reason":             diagnostics.Reason,
		"examined_messages":  diagnostics.ExaminedMessages,
		"retained_messages":  diagnostics.RetainedMessages,
		"protected_messages": diagnostics.ProtectedMessages,
		"removed_messages":   diagnostics.RemovedMessages,
		"removed_turns":      diagnostics.RemovedTurns,
	})
}

// historyResetReplayDiagnostics maps the action's bounded diagnostic onto the
// replay record. Only counts and reason codes cross this boundary.
func historyResetReplayDiagnostics(ctx *RequestContext) *store.HistoryResetDiagnostics {
	if ctx == nil || ctx.HistoryResetDiagnostics == nil {
		return nil
	}
	diagnostics := ctx.HistoryResetDiagnostics
	return &store.HistoryResetDiagnostics{
		Signal:            diagnostics.Signal,
		Scope:             diagnostics.Scope,
		TriggerClass:      string(diagnostics.TriggerClass),
		Version:           diagnostics.Version,
		Outcome:           string(diagnostics.Outcome),
		Reason:            diagnostics.Reason,
		ExaminedMessages:  diagnostics.ExaminedMessages,
		RetainedMessages:  diagnostics.RetainedMessages,
		ProtectedMessages: diagnostics.ProtectedMessages,
		RemovedMessages:   diagnostics.RemovedMessages,
		RemovedTurns:      diagnostics.RemovedTurns,
	}
}

// contextTransformationFailure maps a fail-closed context stage onto the
// response the client receives. A reset that could not be evaluated safely is
// a router-side condition the caller may retry, so it answers 503; an
// invariant the shared executor rejected is an internal fault, and a
// compression-only failure keeps its existing mapping.
func contextTransformationFailure(ctx *RequestContext) (int, string) {
	if ctx == nil || ctx.HistoryResetDiagnostics == nil ||
		ctx.HistoryResetDiagnostics.Outcome != historyreset.OutcomeFailed {
		return 500, "Context compression failed under fail_closed policy"
	}
	if ctx.HistoryResetDiagnostics.Reason == "invariant_violation" {
		return 500, "History reset produced an invalid request under fail_closed policy"
	}
	return 503, "History reset could not be evaluated under fail_closed policy"
}
