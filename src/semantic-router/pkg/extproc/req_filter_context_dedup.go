package extproc

import (
	"maps"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextdedup"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// bindContextDedupPolicy resolves the selected decision's deduplication
// policy once the decision is known. Unlike a reset driven by per-request
// evidence, deduplication is a deterministic, behavior-preserving function of
// the request and the decision's plugin configuration, both of which the
// response cache identity already covers, so an enabled policy does not
// bypass the cache.
func bindContextDedupPolicy(ctx *RequestContext) {
	if ctx == nil || ctx.VSRSelectedDecision == nil {
		return
	}
	policy := ctx.VSRSelectedDecision.GetContextDedupConfig()
	if !policy.IsEnabled() {
		return
	}
	ctx.ContextDedupPolicy = policy
}

// prepareContextDedupStep registers the deduplication step on the request. It
// runs after RAG and Memory enrichment and immediately before the shared
// context stage, so the policy sees the same messages the executor will
// transform. It must not construct the request IR; the shared stage owns
// that, and the resolver reads it only when the step executes.
func (r *OpenAIRouter) prepareContextDedupStep(ctx *RequestContext, request *llmprotocol.Request) {
	if ctx == nil || !ctx.ContextDedupPolicy.IsEnabled() || ctx.ContextDedupAction != nil {
		return
	}
	blocked := ""
	if request == nil || ctx.SemanticRequest == nil {
		blocked = contextdedup.ReasonUnsupportedRepresentation
	}
	action := contextdedup.NewAction(contextDedupPolicy(ctx.ContextDedupPolicy), blocked)
	action.WithResolver(func(id int) (llmprotocol.Message, bool) {
		return contextdedup.RequestResolver(ctx.ContextRequestIR)(id)
	})
	ctx.ContextDedupAction = action
	ctx.ContextHistorySteps = appendContextHistoryStep(ctx.ContextHistorySteps, action.Step())
}

// appendContextHistoryStep inserts a step by kind so the shared plan's
// ascending-order rule holds whatever order the policies registered in.
func appendContextHistoryStep(
	steps []contextcompression.TransformationStep,
	step contextcompression.TransformationStep,
) []contextcompression.TransformationStep {
	position := len(steps)
	for index, existing := range steps {
		if existing.Kind > step.Kind {
			position = index
			break
		}
	}
	steps = append(steps, contextcompression.TransformationStep{})
	copy(steps[position+1:], steps[position:])
	steps[position] = step
	return steps
}

// contextDedupPolicy translates validated configuration into the policy the
// pure action executes with. Configuration cannot widen shared eligibility.
func contextDedupPolicy(configured *config.ContextDedupPluginConfig) contextdedup.Policy {
	limits := configured.EffectiveLimits()
	return contextdedup.Policy{
		Normalization:   contextdedup.Normalization(configured.EffectiveNormalization()),
		MaxHistoryTurns: limits.MaxHistoryTurns,
		MaxHistoryBytes: limits.MaxHistoryBytes,
		MaxSegmentTurns: limits.MaxSegmentTurns,
		FailClosed:      configured.EffectiveFailureMode() == config.ContextDedupFailureClosed,
		Timeout:         time.Duration(limits.TimeoutMs) * time.Millisecond,
	}
}

// finalizeContextDedupDiagnostics reconciles the action's bounded diagnostic
// with the executor's committed receipt. It runs on both the success and the
// failure path so a rejected plan still reports why, and it never records
// more removals than the executor committed.
func finalizeContextDedupDiagnostics(ctx *RequestContext, ir *contextcompression.RequestIR) {
	if ctx == nil || ctx.ContextDedupAction == nil || ir == nil {
		return
	}
	diagnostics := ctx.ContextDedupAction.Reconcile(ir.Transformations.Receipts())
	ctx.ContextDedupDiagnostics = &diagnostics
	metrics.RecordContextDedupEvaluation(
		ctx.VSRSelectedDecisionName,
		string(diagnostics.Outcome),
		diagnostics.Reason,
		diagnostics.RemovedTurns,
		diagnostics.RemovedMessages,
		diagnostics.RemovedTextBytes,
	)
	logging.ComponentDebugEvent("extproc", "context_dedup_evaluated", map[string]interface{}{
		"request_id":         ctx.RequestID,
		"scope":              diagnostics.Scope,
		"normalization":      string(diagnostics.Normalization),
		"outcome":            string(diagnostics.Outcome),
		"reason":             diagnostics.Reason,
		"examined_messages":  diagnostics.ExaminedMessages,
		"examined_turns":     diagnostics.ExaminedTurns,
		"candidate_turns":    diagnostics.CandidateTurns,
		"protected_messages": diagnostics.ProtectedMessages,
		"retained_messages":  diagnostics.RetainedMessages,
		"removed_messages":   diagnostics.RemovedMessages,
		"removed_turns":      diagnostics.RemovedTurns,
		"duplicate_segments": diagnostics.DuplicateSegments,
	})
}

// contextDedupReplayDiagnostics maps the action's bounded diagnostic onto the
// replay record. Only counts, reason codes, and message positions cross this
// boundary.
func contextDedupReplayDiagnostics(ctx *RequestContext) *store.ContextDedupDiagnostics {
	if ctx == nil || ctx.ContextDedupDiagnostics == nil {
		return nil
	}
	diagnostics := ctx.ContextDedupDiagnostics
	record := &store.ContextDedupDiagnostics{
		Scope:             diagnostics.Scope,
		Normalization:     string(diagnostics.Normalization),
		Outcome:           string(diagnostics.Outcome),
		Reason:            diagnostics.Reason,
		ExaminedMessages:  diagnostics.ExaminedMessages,
		ExaminedTurns:     diagnostics.ExaminedTurns,
		CandidateTurns:    diagnostics.CandidateTurns,
		ProtectedMessages: diagnostics.ProtectedMessages,
		RetainedMessages:  diagnostics.RetainedMessages,
		RemovedMessages:   diagnostics.RemovedMessages,
		RemovedTurns:      diagnostics.RemovedTurns,
		RemovedTextBytes:  diagnostics.RemovedTextBytes,
		DuplicateSegments: diagnostics.DuplicateSegments,
		SegmentsTruncated: diagnostics.SegmentsTruncated,
		RecoveryStatus:    diagnostics.RecoveryStatus,
		Recovery:          diagnostics.Recovery,
	}
	if len(diagnostics.Retained) > 0 {
		record.Retained = maps.Clone(diagnostics.Retained)
	}
	for _, segment := range diagnostics.Segments {
		// A struct conversion fails to compile if the two shapes drift.
		record.Segments = append(record.Segments, store.ContextDedupSegment(segment))
	}
	return record
}

// contextTransformationFailure maps a fail-closed context stage onto the
// response the client receives. A deduplication that could not be evaluated
// safely is a router-side condition the caller may retry, so it answers 503;
// an invariant the shared executor rejected is an internal fault, and a
// compression-only failure keeps its existing mapping. A fail-open
// deduplication never stops the plan, so its failed diagnostic cannot be the
// cause of the rejection and the compression mapping stands.
func contextTransformationFailure(ctx *RequestContext) (int, string) {
	if ctx == nil || ctx.ContextDedupDiagnostics == nil ||
		ctx.ContextDedupDiagnostics.Outcome != contextdedup.OutcomeFailed ||
		ctx.ContextDedupPolicy.EffectiveFailureMode() != config.ContextDedupFailureClosed {
		return 500, "Context compression failed under fail_closed policy"
	}
	if ctx.ContextDedupDiagnostics.Reason == "invariant_violation" {
		return 500, "Context deduplication produced an invalid request under fail_closed policy"
	}
	return 503, "Context deduplication could not be evaluated under fail_closed policy"
}
