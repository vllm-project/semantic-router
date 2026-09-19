package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// progressCandidateReason applies the request's candidate boundary and hard
// filters to one exact candidate, so duplicate model names cannot collapse
// into an ambiguous model-only lookup.
func (r *OpenAIRouter) progressCandidateReason(ctx *RequestContext, selCtx *selection.SelectionContext, candidate *config.ModelRef) string {
	if selCtx == nil || candidate == nil || !selectionContextContainsCandidate(selCtx, *candidate) {
		return "candidate_set"
	}
	ref := *candidate
	if !r.configuredBackendModel(ref.Model) {
		return "model_unavailable"
	}
	if ctx == nil {
		return "request_context_missing"
	}
	if r.modelRefExceedsContextWindow(ref, ctx.VSRContextTokenCount) {
		return "context_limit"
	}
	if ctx.SemanticRequest != nil && !r.modelCanServeCapabilities(ref.Model, llmprotocol.RequiredCapabilities(*ctx.SemanticRequest)) {
		return "capability"
	}
	if decision := ctx.VSRSelectedDecision; decision != nil {
		method := r.getSelectionMethod(decision.Algorithm)
		if method == selection.MethodMultiFactor {
			selector, ok := r.selectorForDecisionMethod(method, decision.Algorithm, ctx).(*selection.MultiFactorSelector)
			if !ok || !selector.CandidateEligible(selCtx, ref) {
				return "slo_or_quality"
			}
		}
	}
	return ""
}

// modelCanServeCapabilities reports whether the model can express every
// required capability: the wire format's codec set must cover them, narrowed
// by the model's own declared capabilities when annotated.
func (r *OpenAIRouter) modelCanServeCapabilities(model string, required llmprotocol.CapabilitySet) bool {
	format, err := wireFormatForModel(r.Config.GetModelAPIFormat(model))
	if err != nil {
		return false
	}
	return r.providerCapabilityMismatch(model, format, required) == nil
}

func progressHardLock(selCtx *selection.SelectionContext) string {
	if session := agenticSessionFromContext(selCtx); session != nil {
		if session.ActiveToolLoop {
			return "active_tool_loop"
		}
		if session.HasNonPortableContext {
			return "non_portable_context"
		}
	}
	return ""
}

func (r *OpenAIRouter) progressEligibleContext(ctx *RequestContext, selCtx *selection.SelectionContext) *selection.SelectionContext {
	if selCtx == nil {
		return nil
	}
	out := *selCtx
	out.CandidateModels = make([]config.ModelRef, 0, len(selCtx.CandidateModels))
	for _, ref := range selCtx.CandidateModels {
		if r.progressCandidateReason(ctx, selCtx, &ref) == "" {
			out.CandidateModels = append(out.CandidateModels, ref)
		}
	}
	return &out
}

func selectionContextContainsCandidate(selCtx *selection.SelectionContext, candidate config.ModelRef) bool {
	if selCtx == nil {
		return false
	}
	for _, ref := range selCtx.CandidateModels {
		if selection.CandidateIdentity(ref) == selection.CandidateIdentity(candidate) {
			return true
		}
	}
	return false
}

func attachRejectedRescue(result, rejected *selection.SelectionResult) {
	if result == nil || rejected == nil || rejected.SessionPolicy == nil {
		return
	}
	if result.SessionPolicy == nil {
		result.SessionPolicy = &selection.SessionPolicyTrace{}
	}
	result.SessionPolicy.RescueSwitchGate = rejected.SessionPolicy.SwitchGate
	if trace := result.SessionPolicy.RescueSwitchGate; trace != nil {
		trace.FinalModel = result.SelectedModel
	}
}

func finishProgressTrace(trace *selection.SessionSwitchGateTrace, model string, applied bool, reason string) {
	if trace == nil {
		return
	}
	trace.FinalModel = model
	trace.Applied = applied
	trace.ApplicationReason = reason
}
