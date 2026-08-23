package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

// getReasoningEffort resolves the semantic reasoning level selected for a
// decision. Wire-specific placement belongs exclusively to protocol codecs.
func (r *OpenAIRouter) getReasoningEffort(
	decision *config.Decision,
	modelName string,
) string {
	if r.Config == nil {
		return "medium"
	}
	if decision != nil {
		if effort := r.reasoningEffortForDecision(*decision, modelName); effort != "" {
			return effort
		}
	}
	if r.Config.DefaultReasoningEffort != "" {
		return r.Config.DefaultReasoningEffort
	}
	return "medium"
}

func (r *OpenAIRouter) reasoningEffortForDecision(decision config.Decision, modelName string) string {
	for _, modelRef := range decision.ModelRefs {
		if !r.Config.ModelNameMatches(modelRef.Model, modelName) {
			continue
		}
		return modelRef.ReasoningEffort
	}
	return ""
}

func (r *OpenAIRouter) getModelReasoningFamily(model string) *config.ReasoningFamilyConfig {
	if r.Config == nil {
		return nil
	}
	return r.Config.GetModelReasoningFamily(model)
}
