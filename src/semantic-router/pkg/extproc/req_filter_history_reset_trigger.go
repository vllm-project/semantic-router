package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/historyreset"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/topiccontinuity"
)

// historyResetTrigger resolves the evidence for this request's reset policy.
// The topic-continuity stage has already evaluated every rule the selected
// recipe declares, so the action reads the result of the rule its trigger
// names and never classifies anything itself. A rule that produced no result
// is missing evidence, which the configured failure mode then governs.
//
// A result already present on the request context wins, so an internal
// follow-up that carries its parent's evidence is not re-resolved.
func historyResetTrigger(ctx *RequestContext, policy historyreset.Policy) historyreset.TriggerResult {
	if ctx.HistoryResetTrigger != nil {
		return *ctx.HistoryResetTrigger
	}
	result, ok := ctx.TopicContinuityResults[policy.Signal]
	if !ok {
		return historyreset.TriggerResult{}
	}
	trigger := historyResetTriggerFromTopicContinuity(result, historyResetEvidenceBinding(ctx))
	ctx.HistoryResetTrigger = &trigger
	return trigger
}

// historyResetTriggerFromTopicContinuity maps one topic-continuity result onto
// the action's evidence contract. The result is bound to this request's
// original-history snapshot only when it reports having read that snapshot;
// any other provenance stays unbound, and the action rejects it as stale.
func historyResetTriggerFromTopicContinuity(
	result topiccontinuity.Result,
	snapshotBinding string,
) historyreset.TriggerResult {
	trigger := historyreset.TriggerResult{
		Class:      historyResetTriggerClass(result),
		Confidence: result.Confidence,
		Signal:     result.Signal,
		Version:    result.SchemaVersion,
		Fallback:   result.Fallback,
	}
	if result.HistorySource == topiccontinuity.SourceOriginalSnapshot {
		trigger.Binding = snapshotBinding
	}
	return trigger
}

// historyResetTriggerClass maps the result's class. Conflicting evidence is an
// unknown result with its own reason, kept distinct so diagnostics can tell it
// apart; any class this mapping does not recognize is unknown, which never
// authorizes a removal.
func historyResetTriggerClass(result topiccontinuity.Result) historyreset.TriggerClass {
	switch result.Class {
	case topiccontinuity.ClassChange:
		return historyreset.TriggerChange
	case topiccontinuity.ClassContinuation:
		return historyreset.TriggerContinuation
	}
	if result.Reason == topiccontinuity.ReasonConflicting {
		return historyreset.TriggerConflicting
	}
	return historyreset.TriggerUnknown
}
