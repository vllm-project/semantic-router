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

// EffectiveCandidateDemand previews deterministic policy without changing ingress.
func EffectiveCandidateDemand(request *llmprotocol.Request, decision *config.Decision) (CandidateDemand, error) {
	view, err := EffectiveCandidateRequest(request, decision)
	if err != nil {
		return CandidateDemand{}, err
	}
	return DemandForRequest(view), nil
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
				return &RequestBudgetError{Code: "max_output_tokens_exceeded", Message: fmt.Sprintf("requested output token limit must be between 1 and %d for model %q", params.MaxOutputTokens, model)}
			}
			if *demand.MaxOutputTokens > int64(math.MaxInt) {
				reserve = math.MaxInt
			} else {
				reserve = int(*demand.MaxOutputTokens)
			}
		}
		if llmprotocol.SaturatingTokenSum(demand.InputTokens, reserve) > params.ContextWindowSize {
			return &RequestBudgetError{Code: "context_length_exceeded", Message: fmt.Sprintf("estimated input (%d tokens) plus requested output (%d tokens) exceeds the %d-token context window for model %q", demand.InputTokens, reserve, params.ContextWindowSize, model)}
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
