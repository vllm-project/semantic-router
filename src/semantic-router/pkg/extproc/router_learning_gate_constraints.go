package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// progressCandidateReason reuses the request's candidate boundary and hard filters.
func (r *OpenAIRouter) progressCandidateReason(ctx *RequestContext, selCtx *selection.SelectionContext, model string) string {
	if selCtx == nil || !selectionContextContainsModel(selCtx, model) {
		return "candidate_set"
	}
	ref := selectedModelRefFromResult(selCtx, &selection.SelectionResult{SelectedModel: model})
	if ref == nil || !r.configuredBackendModel(ref.Model) {
		return "model_unavailable"
	}
	if ctx == nil {
		return "request_context_missing"
	}
	if r.modelRefExceedsContextWindow(*ref, ctx.VSRContextTokenCount) {
		return "context_limit"
	}
	if ctx.SemanticRequest != nil && r.qualifiedRerouteCandidate(ref.Model, llmprotocol.RequiredCapabilities(*ctx.SemanticRequest)) == "" {
		return "capability"
	}
	if decision := ctx.VSRSelectedDecision; decision != nil {
		method := r.getSelectionMethod(decision.Algorithm)
		if method == selection.MethodMultiFactor {
			selector, ok := r.selectorForDecisionMethod(method, decision.Algorithm, ctx).(*selection.MultiFactorSelector)
			if !ok || !selector.CandidateEligible(selCtx, ref.Model) {
				return "slo_or_quality"
			}
		}
	}
	return ""
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
		if r.progressCandidateReason(ctx, selCtx, ref.Model) == "" {
			out.CandidateModels = append(out.CandidateModels, ref)
		}
	}
	return &out
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
