package extproc

import (
	"errors"
	"fmt"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/inflight"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func (r *OpenAIRouter) prepareAdmittedDispatch(dispatch *providerDispatch, ctx *RequestContext) (*preparedProviderRequest, dispatchAdmission, error) {
	if ctx.SemanticRequest == nil {
		prepared, err := r.materializePreparedProviderRequest(dispatch, ctx)
		return prepared, dispatchAdmission{}, err
	}
	start := len(ctx.ProtocolDiagnostics)
	ctx.projectionObservations = new([]func())
	cacheAction, cacheReason := ctx.PromptCacheAction, ctx.PromptCacheReason
	cacheInserted, cachePreserved := ctx.PromptCacheInserted, ctx.PromptCachePreserved
	prepared, admission, err := r.prepareFinalCandidate(dispatch, ctx, true)
	if !isFinalContextOverflow(err) {
		return prepared, admission, err
	}
	admission.ExcludedCandidates = 1
	if dispatch.entrypoint == "" || ctx.VSRSelectedDecision == nil {
		return prepared, admission, err
	}
	if policy, ok := protectionLearningPolicyForContext(ctx); ok && policy.Mode == config.DecisionAdaptationModeApply && policy.HardLocked() {
		return nil, admission, err
	}
	if pending := ctx.pendingSessionDecision; pending != nil && (pending.ActiveToolLoop || pending.Policy["hard_locked"] == true) {
		return nil, admission, err
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
		useReasoning := r.candidateReasoningChoice(&view, model)
		alternate, resolveErr := r.resolveProviderDispatchForCandidate(model, dispatch.decisionName, useReasoning, &view)
		if resolveErr != nil {
			return nil, admission, resolveErr
		}
		request, cloneErr := cloneSemanticRequestForReplay(ctx.SemanticRequest)
		if cloneErr != nil {
			return nil, admission, cloneErr
		}
		request.Model = alternate.upstreamModel
		request.Generation++
		view.SemanticRequest = request
		view.RequestModel, view.VSRSelectedModel = model, model
		view.TargetFormat, view.ResponseVendor = alternate.targetFormat, resolveResponseVendor(alternate.profile)
		view.primaryBackendName = alternate.backendName
		view.structuredOutputPass = nil
		request.ReasoningMode, request.ReasoningEffort = dispatch.callerMode, dispatch.callerEffort
		request.ReasoningDisplay, request.ReasoningBudgetTokens = dispatch.callerDisplay, dispatch.callerBudget
		if alternate.targetFormat != llmprotocol.OpenAIChatV1 {
			r.applySemanticReasoningMode(request, alternate.effectiveBackendModel(), alternate.targetFormat, useReasoning, view.decisionForBackend(model))
		}
		r.trackVSRDecision(&view, ctx.VSRSelectedCategory, dispatch.decisionName, model, useReasoning)
		candidateRequest, candidateAdmission, candidateErr := r.prepareFinalCandidate(alternate, &view, false)
		candidateAdmission.ExcludedCandidates = admission.ExcludedCandidates
		if isFinalContextOverflow(candidateErr) {
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
			return nil, candidateAdmission, candidateErr
		}
		ctx.projectionObservations = view.projectionObservations
		ctx.VSRSelectedCandidate, ctx.VSRSelectedModel = &ref, model
		ctx.RequestModel, ctx.SemanticRequest = model, request
		ctx.TargetFormat, ctx.ResponseVendor = view.TargetFormat, view.ResponseVendor
		ctx.primaryBackendName, ctx.PluginBackendModel = alternate.backendName, model
		ctx.structuredOutputPass, ctx.ProtocolDiagnostics = view.structuredOutputPass, view.ProtocolDiagnostics
		ctx.VSRReasoningMode = view.VSRReasoningMode
		ctx.VSRSelectionReasoning = "Final context admission selected an alternate candidate"
		ctx.PromptCacheAction, ctx.PromptCacheReason = view.PromptCacheAction, view.PromptCacheReason
		ctx.PromptCacheInserted, ctx.PromptCachePreserved = view.PromptCacheInserted, view.PromptCachePreserved
		if ctx.CacheSelectedModel != "" {
			ctx.CacheSelectedModel = model
			ctx.CacheIdentity = responseCacheIdentity(ctx, ctx.CacheRequestModel)
		}
		if pending := ctx.pendingSessionDecision; pending != nil {
			pending.SelectedModel, pending.SelectedCandidate = ref.Model, &ref
		}
		if ctx.InflightToken != 0 {
			inflight.End(ctx.InflightModel, ctx.InflightToken)
			ctx.InflightModel, ctx.InflightToken = model, inflight.Begin(model)
		}
		alternate.entrypoint = dispatch.entrypoint
		*dispatch = *alternate
		return candidateRequest, candidateAdmission, nil
	}
	if incompatible != nil {
		return nil, admission, incompatible
	}
	return nil, admission, err
}
