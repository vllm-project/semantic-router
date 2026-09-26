package extproc

import (
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// prepareModelInput materializes backend-independent content. Both operations
// are idempotent: dispatch does not prepend retained history or read image files
// again when selection already prepared this request.
func (r *OpenAIRouter) prepareModelInput(request *llmprotocol.Request, ctx *RequestContext) (bool, error) {
	changed, err := r.materializeResponseObjectContext(request, ctx)
	if err != nil {
		return changed, err
	}
	inlined, err := r.resolveImageFileReferences(request)
	return changed || inlined, err
}

// selectionCapabilityRequest previews the decision's tool policy without running
// plugins twice. Candidate-specific reasoning and request parameters are projected
// on a separate value below. The live request retains its original tool history.
func (r *OpenAIRouter) selectionCapabilityRequest(ctx *RequestContext) (*llmprotocol.Request, error) {
	if ctx == nil || ctx.SemanticRequest == nil {
		return nil, nil
	}
	changed, err := r.prepareModelInput(ctx.SemanticRequest, ctx)
	if err != nil {
		return nil, err
	}
	if changed {
		ctx.SemanticRequest.Generation++
	}
	request := *ctx.SemanticRequest
	request.Stream = ctx.ExpectStreamingResponse
	if tools := resolveDecisionToolsConfig(ctx); tools != nil && tools.Enabled && tools.EffectiveMode() == config.ToolsPluginModeNone {
		if tools.StripToolHistory {
			request.Messages = cloneMessagesForToolStrip(request.Messages)
		}
		stripSemanticToolPolicy(&request, tools.StripToolHistory)
	}
	return &request, nil
}

// candidateCapabilityMismatch uses the same capability contract as final
// dispatch. Unannotated models retain the existing wire-only qualification.
func (r *OpenAIRouter) candidateCapabilityMismatch(ref config.ModelRef, request *llmprotocol.Request, decision *config.Decision, requirements *config.CandidateRequirements, automaticDemands map[string]selection.CandidateDemand) error {
	if request == nil {
		return nil // Eval/selector-only callers have no full inference envelope.
	}
	format, err := wireFormatForModel(r.Config.GetModelAPIFormat(ref.Model))
	if err != nil {
		return err
	}
	preview := *request
	if decision != nil && !preserveExplicitAnthropicReasoning(&preview, format) {
		if format != llmprotocol.OpenAIChatV1 {
			if family := r.getModelReasoningFamily(ref.Model); family != nil {
				exact := *decision
				exact.ModelRefs = []config.ModelRef{ref}
				enabled := ref.UseReasoning != nil && *ref.UseReasoning
				preview.ReasoningEffort, preview.ReasoningMode = semanticReasoningControls(r, family, &exact, ref.Model, format, enabled)
				if format != llmprotocol.AnthropicMessagesV1 || preview.ReasoningMode != llmprotocol.ReasoningModeEnabled {
					preview.ReasoningBudgetTokens = nil
				}
			}
		}
	}
	preview, err = r.projectAnthropicRequestForBackend(preview, ref.Model, format)
	if err != nil {
		return err
	}
	demand, err := selection.EffectiveCandidateDemand(&preview, decision)
	if err != nil {
		return err
	}
	if demand.AutomaticOutput {
		// Selection already rendered this candidate without mutating ingress.
		// Keep freshly projected capabilities, but qualify the exact rendered
		// budget instead of rejecting the intentionally absent caller limit.
		rendered, ok := automaticDemands[ref.Model]
		if !ok || !rendered.Known || !rendered.AutomaticOutput || rendered.InputTokens < 0 || rendered.MaxOutputTokens == nil || *rendered.MaxOutputTokens <= 0 {
			return automaticOutputUnsupported("a resolved candidate budget before capability selection")
		}
		demand.InputTokens = rendered.InputTokens
		demand.MaxOutputTokens = rendered.MaxOutputTokens
	}
	if err := r.validateModelDemand(requirements, ref.Model, demand); err != nil {
		return err
	}
	return r.providerCapabilityMismatch(ref.Model, format, demand.Capabilities)
}

func (r *OpenAIRouter) capabilityEligibleSelectionContext(input *selection.SelectionContext, algorithm *config.AlgorithmConfig, ctx *RequestContext) (*selection.SelectionContext, error) {
	if r == nil || r.Config == nil || ctx == nil || ctx.SemanticRequest == nil ||
		(algorithm != nil && config.IsLooperAlgorithmType(algorithm.Type)) {
		return input, nil
	}
	if err := selection.ValidateSelectionContext(input); err != nil {
		return nil, err
	}
	request, err := r.selectionCapabilityRequest(ctx)
	if err != nil {
		return nil, err
	}
	requirements := r.candidateRequirements(ctx)
	eligible := make([]config.ModelRef, 0, len(input.CandidateModels))
	var unsupported *llmprotocol.ProtocolError
	onlyUnsupported := true
	// Context, budget, and hard-policy filters can narrow the decision before
	// this stage. A wire mismatch in the remainder is not the sole exclusion.
	if ctx.VSRSelectedDecision != nil {
		for _, ref := range ctx.VSRSelectedDecision.ModelRefs {
			if !modelRefInEligibility(ref, input.CandidateModels) {
				onlyUnsupported = false
				break
			}
		}
	}
	for _, ref := range input.CandidateModels {
		if (!decisionUsesAutomaticOutput(request, ctx.VSRSelectedDecision) && !selection.CandidateRequirementsEnabled(requirements) && r.modelRefExceedsContextWindow(ref, ctx.VSRContextTokenCount)) ||
			(ctx.VSREligibleModelRefs != nil && !modelRefInEligibility(ref, ctx.VSREligibleModelRefs)) ||
			(ctx.VSRPolicyEligibleModelRefs != nil && !modelRefInEligibility(ref, ctx.VSRPolicyEligibleModelRefs)) {
			onlyUnsupported = false
			continue
		}
		if err := r.candidateCapabilityMismatch(ref, request, ctx.VSRSelectedDecision, requirements, ctx.AutomaticCandidateDemands); err != nil {
			var protocolErr *llmprotocol.ProtocolError
			if !errors.As(err, &protocolErr) || protocolErr.Category != llmprotocol.ErrorUnsupportedFeature {
				onlyUnsupported = false
			} else if unsupported == nil {
				unsupported = protocolErr
			}
			continue
		}
		eligible = append(eligible, ref)
	}
	if len(eligible) == 0 {
		// Dispatch to a named model reports this mismatch as a client error, so
		// auto routing does too when it is the only reason no candidate qualified.
		if onlyUnsupported && unsupported != nil {
			return nil, unsupported
		}
		return nil, fmt.Errorf("%w: decision %q has no candidate supporting the request capabilities", selection.ErrNoEligibleCandidates, input.DecisionName)
	}
	decision := &config.Decision{Name: input.DecisionName, Algorithm: algorithm}
	if err := validateMinimumEligibleDecisionModels(decision, eligible, ctx.VSRContextTokenCount); err != nil {
		return nil, fmt.Errorf("%w: %s", selection.ErrNoEligibleCandidates, err.Error())
	}
	ctx.VSREligibleModelRefs = cloneModelRefs(eligible)
	filtered := *input
	filtered.CandidateModels = eligible
	return &filtered, nil
}
