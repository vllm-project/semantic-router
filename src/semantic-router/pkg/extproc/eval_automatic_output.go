package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

// Native Preview reuses candidate rendering, never the live preparation chain:
// that chain may compress input, record a selection or update session state.
func (r *OpenAIRouter) selectAutomaticEvalCandidate(input services.EvalModelSelectionInput) services.EvalModelSelection {
	decision := input.Decision
	method := r.getSelectionMethod(decision.Algorithm)
	if config.IsLooperAlgorithmType(evalAlgorithmType(decision)) || !evalSupportsDryRunSelection(method) {
		return automaticEvalExecutionRequired(decision, "the configured algorithm requires execution")
	}
	effective, err := selection.EffectiveCandidateRequest(input.SemanticRequest, decision)
	if err != nil {
		return evalSelectionUnavailable(err.Error())
	}
	if reason := automaticEvalRequestLimitation(effective, decision); reason != "" {
		return automaticEvalExecutionRequired(decision, reason)
	}
	ctx, err := r.prepareEvalRequest(input, decision)
	if err != nil {
		return evalSelectionUnavailable(err.Error())
	}
	refs := decision.ModelRefs
	routeAction := decision.Action != nil && decision.Action.Type == config.DecisionActionRoute && strings.TrimSpace(decision.Action.Destination) != ""
	if routeAction {
		destination := strings.TrimSpace(decision.Action.Destination)
		ref := config.ModelRef{Model: destination}
		for _, configured := range refs {
			if configured.Model == destination {
				ref = configured
				break
			}
		}
		refs = []config.ModelRef{ref}
	}
	for _, ref := range refs {
		if ref.LoRAName != "" {
			return automaticEvalExecutionRequired(decision, "automatic output preview does not resolve LoRA requests")
		}
	}
	demands, overflows, err := r.renderAutomaticCandidates(ctx, refs)
	if err != nil {
		return evalSelectionUnavailable(err.Error())
	}
	if err := input.Context.Err(); err != nil {
		return evalSelectionUnavailable(err.Error())
	}
	if len(demands) == 0 {
		if len(overflows) > 0 && contextOverflowConfig(ctx) != nil {
			return automaticEvalExecutionRequired(decision, "the rendered input requires configured context compression")
		}
		return evalSelectionUnavailable("no candidate has a compatible provider-rendered output budget")
	}
	ctx.AutomaticCandidateDemands = demands
	eligible := make([]config.ModelRef, 0, len(refs))
	for _, ref := range refs {
		if _, ok := demands[ref.Model]; ok {
			eligible = append(eligible, ref)
		}
	}
	ctx.VSREligibleModelRefs = cloneModelRefs(eligible)
	ctx.VSRPolicyEligibleModelRefs = cloneModelRefs(eligible)
	if routeAction {
		return services.EvalModelSelection{SelectedModel: eligible[0].Model, Status: services.EvalSelectionSelected, Method: "route_action", Reason: "declared route action satisfying the provider-rendered request"}
	}
	filtered := *decision
	filtered.ModelRefs = eligible
	if err := validateMinimumEligibleDecisionModels(&filtered, eligible, input.ContextTokenCount); err != nil {
		return evalSelectionUnavailable(err.Error())
	}
	ctx.VSRSelectedDecision = &filtered
	return r.selectEvalCandidate(input, &filtered, method, ctx)
}

func automaticEvalExecutionRequired(decision *config.Decision, reason string) services.EvalModelSelection {
	return services.EvalModelSelection{Status: services.EvalSelectionExecutionRequired, Method: evalAlgorithmType(decision), Reason: reason}
}

// The public Preview envelope is prompt-bearing, not an arbitrary generation
// request. Only deterministic projections already shared with admission are
// supported; runtime enrichment must not silently disappear from its budget.
func automaticEvalRequestLimitation(request *llmprotocol.Request, decision *config.Decision) string {
	if request == nil {
		return "automatic output preview requires the prompt-bearing request"
	}
	required := llmprotocol.RequiredCapabilities(*request)
	for _, capability := range []llmprotocol.Capability{
		llmprotocol.CapabilityImageInput, llmprotocol.CapabilityAudioInput,
		llmprotocol.CapabilityVideoInput, llmprotocol.CapabilityFileInput,
		llmprotocol.CapabilityImageGeneration, llmprotocol.CapabilityConversationState,
		llmprotocol.CapabilityReasoningSignature,
	} {
		if required.Supports(capability) {
			return "automatic output preview requires text and inline tool content without external or opaque provider state"
		}
	}
	if rag := decision.GetRAGConfig(); rag != nil && rag.Enabled {
		return "retrieval enrichment requires execution"
	}
	if memory := decision.GetMemoryConfig(); memory != nil && memory.Enabled {
		return "memory enrichment requires execution"
	}
	tools := decision.GetToolsConfig()
	if tools != nil && tools.Enabled && tools.EffectiveMode() == config.ToolsPluginModeFiltered {
		return "configured tool filtering requires execution"
	}
	if request.ToolChoice.Mode == llmprotocol.ToolChoiceAuto && (tools == nil || !tools.Enabled || tools.EffectiveMode() != config.ToolsPluginModeNone) {
		selectionPlugin := decision.GetToolSelectionConfig()
		if tools.SelectionEnabled() || selectionPlugin != nil && selectionPlugin.Enabled {
			return "semantic tool selection requires execution"
		}
	}
	if compression := decision.GetContextCompressionConfig(); compression != nil && compression.Enabled {
		if compression.Targets != nil && compression.EffectiveTargetMode(compression.Targets.History) != config.ContextCompressionTargetPreserve && len(request.Messages) > 1 {
			return "configured history compression is not resolved by automatic output preview"
		}
		if compression.EffectiveToolOutputTarget().Mode != config.ContextCompressionTargetPreserve {
			for _, message := range request.Messages {
				if message.Role == llmprotocol.RoleTool {
					return "configured tool-output compression is not resolved by automatic output preview"
				}
			}
		}
	}
	return ""
}
