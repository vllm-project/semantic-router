package selection

import (
	"fmt"
	"math"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// CandidateDemand contains only request facts, never messages or tool schemas.
// InputTokens is an estimate and excludes the output reserve.
type CandidateDemand struct {
	Known             bool
	Capabilities      llmprotocol.CapabilitySet
	ModelCapabilities llmprotocol.CapabilitySet
	InputTokens       int
	MaxOutputTokens   *int64
}

func DemandForRequest(request *llmprotocol.Request) CandidateDemand {
	if request == nil {
		return CandidateDemand{}
	}
	required := llmprotocol.RequiredCapabilities(*request)
	modelMask := llmprotocol.Capabilities(
		llmprotocol.CapabilityText, llmprotocol.CapabilityTools,
		llmprotocol.CapabilityStructuredJSON, llmprotocol.CapabilityStrictJSONSchema,
	)
	modelRequired := required.TaskCapabilities()
	// Model declarations and transport fidelity have different semantics:
	// parallel tool framing, for example, does not require a separate model tag.
	modelNames := append(modelRequired.Names(), required.Intersect(modelMask).Names()...)
	if required.Supports(llmprotocol.CapabilityStrictJSONSchema) {
		modelNames = append(modelNames, "structured_json")
	}
	if request.ReasoningMode != llmprotocol.ReasoningModeDisabled &&
		(request.ReasoningMode == llmprotocol.ReasoningModeEnabled || request.ReasoningMode == llmprotocol.ReasoningModeAdaptive ||
			(request.ReasoningEffort != "" && !strings.EqualFold(strings.TrimSpace(request.ReasoningEffort), "none")) || request.ReasoningBudgetTokens != nil) {
		modelNames = append(modelNames, "reasoning")
	}

	// Strictness is enforced by the codec; model metadata uses structured_json.
	filtered := modelNames[:0]
	for _, name := range modelNames {
		if name != "strict_json_schema" {
			filtered = append(filtered, name)
		}
	}
	modelRequired, _ = llmprotocol.ParseCapabilities(filtered)
	demand := CandidateDemand{Known: true, Capabilities: required, ModelCapabilities: modelRequired, InputTokens: llmprotocol.EstimateInput(request).Tokens}
	if request.Sampling.MaxOutputTokens != nil {
		value := *request.Sampling.MaxOutputTokens
		demand.MaxOutputTokens = &value
	}
	return demand
}

// EffectiveCandidateDemand previews only deterministic decision mutations. The
// ingress request stays untouched for signals, retention and actual plugins.
func EffectiveCandidateDemand(request *llmprotocol.Request, decision *config.Decision) (CandidateDemand, error) {
	if request == nil {
		return CandidateDemand{}, nil
	}
	view := *request
	view.Instructions = append([]llmprotocol.InstructionBlock(nil), request.Instructions...)
	view.Messages = append([]llmprotocol.Message(nil), request.Messages...)
	for i := range view.Messages {
		view.Messages[i].Content = append([]llmprotocol.Content(nil), request.Messages[i].Content...)
	}
	if decision != nil {
		if prompt := decision.GetSystemPromptConfig(); prompt != nil && decision.IsSystemPromptEnabled() {
			llmprotocol.SetSystemInstruction(&view, prompt.SystemPrompt, decision.GetSystemPromptMode())
		}
		if tools := decision.GetToolsConfig(); tools != nil && tools.Enabled && tools.EffectiveMode() == config.ToolsPluginModeNone {
			llmprotocol.StripTools(&view, tools.StripToolHistory)
		}
		if params := decision.GetRequestParamsConfig(); params != nil {
			for _, field := range params.BlockedParams {
				if _, err := llmprotocol.BlockRequestField(&view, strings.TrimSpace(field)); err != nil {
					return CandidateDemand{}, err
				}
			}
			llmprotocol.DefaultOutputTokens(&view, params.DefaultMaxTokens)
			llmprotocol.CapOutputTokens(&view, params.MaxTokensLimit)
			llmprotocol.CapCandidateCount(&view, params.MaxN)
		}
	}
	return DemandForRequest(&view), nil
}

func CandidateRequirementsEnabled(requirements *config.CandidateRequirements) bool {
	return requirements != nil && (requirements.Capabilities != "" || requirements.Context != "")
}

// ValidateCandidateRequirements is shared by live selection, Preview, and
// individual Looper calls. Missing required metadata excludes a candidate.
func ValidateCandidateRequirements(requirements *config.CandidateRequirements, model string, params config.ModelParams, demand CandidateDemand) error {
	if requirements == nil {
		return nil
	}
	if CandidateRequirementsEnabled(requirements) && !demand.Known {
		return fmt.Errorf("%w: request capability and budget facts are unavailable", ErrNoEligibleCandidates)
	}
	if requirements.Capabilities == config.CandidateCapabilitiesDeclared {
		declared, known := llmprotocol.ModelCapabilities(params.Capabilities)
		if !known || !declared.Contains(demand.ModelCapabilities) {
			return fmt.Errorf("%w: model %q lacks declared request capabilities", ErrNoEligibleCandidates, model)
		}
	}
	if requirements.Context == config.CandidateContextKnownLimits {
		if demand.InputTokens < 0 {
			return fmt.Errorf("%w: request input estimate is invalid", ErrNoEligibleCandidates)
		}
		if params.ContextWindowSize <= 0 || params.MaxOutputTokens <= 0 {
			return fmt.Errorf("%w: model %q lacks known context or output limits", ErrNoEligibleCandidates, model)
		}
		if demand.MaxOutputTokens == nil {
			return fmt.Errorf("%w: an explicit caller or request policy output limit is required", ErrNoEligibleCandidates)
		}
		reserve := 0
		if demand.MaxOutputTokens != nil {
			if *demand.MaxOutputTokens <= 0 || *demand.MaxOutputTokens > int64(params.MaxOutputTokens) {
				return fmt.Errorf("%w: model %q cannot satisfy the requested output limit", ErrNoEligibleCandidates, model)
			}
			if *demand.MaxOutputTokens > int64(math.MaxInt) {
				reserve = math.MaxInt
			} else {
				reserve = int(*demand.MaxOutputTokens)
			}
		}
		if llmprotocol.SaturatingTokenSum(demand.InputTokens, reserve) > params.ContextWindowSize {
			return fmt.Errorf("%w: model %q cannot satisfy the estimated input plus output budget", ErrNoEligibleCandidates, model)
		}
	}
	return nil
}

func ValidateCandidateCodec(requirements *config.CandidateRequirements, model string, supported llmprotocol.CapabilitySet, demand CandidateDemand) error {
	if requirements != nil && requirements.Capabilities == config.CandidateCapabilitiesDeclared && !supported.Contains(demand.Capabilities) {
		return fmt.Errorf("%w: model %q codec cannot preserve the request", ErrNoEligibleCandidates, model)
	}
	return nil
}
