package extproc

import (
	"errors"
	"fmt"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/inflight"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/pluginruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

type dispatchCandidate struct {
	request   *preparedProviderRequest
	alternate *providerDispatch
	view      *RequestContext
}

func (r *OpenAIRouter) prepareAdmittedDispatch(dispatch *providerDispatch, ctx *RequestContext) (dispatchCandidate, dispatchAdmission, error) {
	if ctx.SemanticRequest == nil {
		prepared, err := r.materializePreparedProviderRequest(dispatch, ctx)
		return dispatchCandidate{request: prepared}, dispatchAdmission{}, err
	}
	start := len(ctx.ProtocolDiagnostics)
	ctx.projectionObservations = new([]func())
	cacheAction, cacheReason := ctx.PromptCacheAction, ctx.PromptCacheReason
	cacheInserted, cachePreserved := ctx.PromptCacheInserted, ctx.PromptCachePreserved
	prepared, admission, err := r.prepareFinalCandidate(dispatch, ctx, true)
	if !isDispatchBudgetError(err) {
		return dispatchCandidate{request: prepared}, admission, err
	}
	if cancellation := selectionRequestContext(ctx).Err(); cancellation != nil {
		return dispatchCandidate{request: prepared}, admission, cancellation
	}
	admission.ExcludedCandidates = 1
	if dispatch.entrypoint == "" || ctx.VSRSelectedDecision == nil || finalModelLocked(ctx) {
		return dispatchCandidate{request: prepared}, admission, err
	}
	var incompatible error
	for _, ref := range ctx.dispatchSelection.OrderCandidates(ctx.VSREligibleModelRefs) {
		model := candidateModelIdentity(ref)
		isPrimary := model == dispatch.logicalModel && (ctx.VSRSelectedCandidate == nil ||
			selection.CandidateIdentity(ref) == selection.CandidateIdentity(*ctx.VSRSelectedCandidate))
		if isPrimary || !modelRefInEligibility(ref, ctx.VSRSelectedDecision.ModelRefs) ||
			ctx.VSRPolicyEligibleModelRefs != nil && !modelRefInEligibility(ref, ctx.VSRPolicyEligibleModelRefs) {
			continue
		}
		view := *ctx
		view.projectionObservations = new([]func())
		view.PromptCacheAction, view.PromptCacheReason = cacheAction, cacheReason
		view.PromptCacheInserted, view.PromptCachePreserved = cacheInserted, cachePreserved
		view.ProtocolDiagnostics = slices.Clone(ctx.ProtocolDiagnostics[:start])
		view.VSRSelectedCandidate = &ref
		useReasoning := r.decisionReasoningChoice(&view, model)
		if ref.UseReasoning != nil {
			useReasoning = *ref.UseReasoning
		}
		view.VSRReasoningReason = ""
		if useReasoning {
			view.VSRReasoningReason = r.unstatedReasoningOffReason(&view, view.backendModelForCandidate(model))
			useReasoning = view.VSRReasoningReason == ""
		}
		alternate, resolveErr := r.resolveProviderDispatchForCandidate(model, dispatch.decisionName, useReasoning, &view)
		if resolveErr != nil {
			return dispatchCandidate{}, admission, resolveErr
		}
		request, cloneErr := cloneSemanticRequestForReplay(ctx.SemanticRequest)
		if cloneErr != nil {
			return dispatchCandidate{}, admission, cloneErr
		}
		request.Model = alternate.upstreamModel
		request.Generation++
		bindProviderRequest(&view, alternate, request)
		view.structuredOutputPass = nil
		request.ReasoningMode, request.ReasoningEffort = dispatch.callerMode, dispatch.callerEffort
		request.ReasoningDisplay, request.ReasoningBudgetTokens = dispatch.callerDisplay, dispatch.callerBudget
		request.Sampling.MaxOutputTokens = dispatch.callerOutputLimit
		r.applyDispatchReasoning(request, alternate, &view)
		if _, policyErr := pluginruntime.ApplyRequestParams(request, r.dispatchRequestParams(&view)); policyErr != nil {
			return dispatchCandidate{}, admission, policyErr
		}
		r.trackVSRDecision(&view, ctx.VSRSelectedCategory, dispatch.decisionName, model, useReasoning)
		if explicitAnthropicReasoningDisabled(&view, alternate) {
			view.VSRReasoningMode = "off"
		}
		candidateRequest, candidateAdmission, candidateErr := r.prepareFinalCandidate(alternate, &view, false)
		candidateAdmission.ExcludedCandidates = admission.ExcludedCandidates
		if isDispatchBudgetError(candidateErr) {
			candidateAdmission.ExcludedCandidates++
			admission, err = candidateAdmission, candidateErr
			continue
		}
		if candidateErr != nil {
			var protocolError *llmprotocol.ProtocolError
			if errors.Is(candidateErr, selection.ErrNoEligibleCandidates) || errors.As(candidateErr, &protocolError) && protocolError.Category == llmprotocol.ErrorUnsupportedFeature {
				incompatible = fmt.Errorf("%w: an alternate cannot serve the final request", selection.ErrNoEligibleCandidates)
				continue
			}
			return dispatchCandidate{}, candidateAdmission, candidateErr
		}
		alternate.entrypoint = dispatch.entrypoint
		return dispatchCandidate{request: candidateRequest, alternate: alternate, view: &view}, candidateAdmission, nil
	}
	if incompatible != nil {
		return dispatchCandidate{}, admission, incompatible
	}
	return dispatchCandidate{}, admission, err
}

func finalModelLocked(ctx *RequestContext) bool {
	if policy, ok := protectionLearningPolicyForContext(ctx); ok && policy.Mode == config.DecisionAdaptationModeApply && policy.HardLocked() {
		return true
	}
	pending := ctx.pendingSessionDecision
	return pending != nil && (pending.ActiveToolLoop || pending.Policy["hard_locked"] == true)
}

func commitDispatchCandidate(ctx *RequestContext, dispatch *providerDispatch, candidate dispatchCandidate) {
	view, alternate := candidate.view, candidate.alternate
	model := alternate.logicalModel
	ctx.projectionObservations = view.projectionObservations
	ctx.VSRSelectedCandidate = view.VSRSelectedCandidate
	bindProviderRequest(ctx, alternate, view.SemanticRequest)
	ctx.PluginBackendModel = model
	ctx.structuredOutputPass, ctx.ProtocolDiagnostics = view.structuredOutputPass, view.ProtocolDiagnostics
	ctx.VSRReasoningMode, ctx.VSRReasoningReason = view.VSRReasoningMode, view.VSRReasoningReason
	ctx.VSRSelectionReasoning = "Final context admission selected an alternate candidate"
	ctx.PromptCacheAction, ctx.PromptCacheReason = view.PromptCacheAction, view.PromptCacheReason
	ctx.PromptCacheInserted, ctx.PromptCachePreserved = view.PromptCacheInserted, view.PromptCachePreserved
	if pending := ctx.pendingSessionDecision; pending != nil {
		pending.SelectedModel, pending.SelectedCandidate = view.VSRSelectedCandidate.Model, view.VSRSelectedCandidate
	}
	if ctx.InflightToken != 0 && ctx.InflightModel != model {
		inflight.End(ctx.InflightModel, ctx.InflightToken)
		ctx.InflightModel, ctx.InflightToken = model, inflight.Begin(model)
	}
	*dispatch = *alternate
}
