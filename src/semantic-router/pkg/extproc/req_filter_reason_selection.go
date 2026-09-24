package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// getReasoningEffort returns the reasoning effort level for a given decision and model.
func (r *OpenAIRouter) getReasoningEffort(decision *config.Decision, modelName string) string {
	if r.Config == nil {
		return "medium"
	}
	if decision != nil {
		if effort := r.reasoningEffortForDecision(*decision, modelName); effort != "" {
			if family := r.getModelReasoningFamily(modelName); reasoningFamilyAllowsEffort(family, effort) {
				return effort
			}
		}
	}
	family := r.getModelReasoningFamily(modelName)
	if r.Config.DefaultReasoningEffort != "" && reasoningFamilyAllowsEffort(family, r.Config.DefaultReasoningEffort) {
		return r.Config.DefaultReasoningEffort
	}
	if family != nil {
		return family.Default
	}
	return "medium"
}

func (r *OpenAIRouter) getReasoningMode(decision *config.Decision, modelName string, enabled bool) string {
	family := r.getModelReasoningFamily(modelName)
	if family == nil {
		return ""
	}
	if !enabled {
		if reasoningFamilySupportsMode(family, string(llmprotocol.ReasoningModeDisabled)) {
			return string(llmprotocol.ReasoningModeDisabled)
		}
		return ""
	}
	if decision != nil {
		if mode := r.reasoningModeForDecision(*decision, modelName); mode != "" &&
			reasoningFamilySupportsMode(family, mode) {
			return mode
		}
	}
	if family.DefaultMode != "" && family.DefaultMode != string(llmprotocol.ReasoningModeDisabled) {
		return family.DefaultMode
	}
	if reasoningFamilySupportsMode(family, string(llmprotocol.ReasoningModeAdaptive)) {
		return string(llmprotocol.ReasoningModeAdaptive)
	}
	if reasoningFamilySupportsMode(family, string(llmprotocol.ReasoningModeEnabled)) {
		return string(llmprotocol.ReasoningModeEnabled)
	}
	return ""
}

func reasoningFamilyAllowsEffort(family *config.ReasoningFamilyConfig, effort string) bool {
	if family == nil || len(family.Levels) == 0 {
		return true
	}
	for _, supported := range family.Levels {
		if effort == supported {
			return true
		}
	}
	return false
}

// modelRefMatchesCandidate reports whether ref matches the candidate identity.
// For a LoRA adapter (LoRAName != ""), it matches only against LoRAName (or its alias).
// For a bare base model (LoRAName == ""), it matches against Model (or its alias).
func (r *OpenAIRouter) modelRefMatchesCandidate(ref config.ModelRef, candidateIdentity string) bool {
	identity := candidateModelIdentity(ref)
	if r != nil && r.Config != nil {
		return r.Config.ModelNameMatches(identity, candidateIdentity)
	}
	return identity == candidateIdentity
}

func (r *OpenAIRouter) reasoningEffortForDecision(decision config.Decision, modelName string) string {
	for _, modelRef := range decision.ModelRefs {
		if r.modelRefMatchesCandidate(modelRef, modelName) {
			return modelRef.ReasoningEffort
		}
	}
	return ""
}

func (r *OpenAIRouter) reasoningModeForDecision(decision config.Decision, modelName string) string {
	for _, modelRef := range decision.ModelRefs {
		if r.modelRefMatchesCandidate(modelRef, modelName) {
			return modelRef.ReasoningMode
		}
	}
	return ""
}

func (r *OpenAIRouter) getModelReasoningFamily(model string) *config.ReasoningFamilyConfig {
	if r.Config == nil {
		return nil
	}
	return r.Config.GetModelReasoningFamily(model)
}
