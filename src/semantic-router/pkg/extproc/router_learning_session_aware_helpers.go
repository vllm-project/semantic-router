package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func currentLearningModel(selCtx *selection.SelectionContext) string {
	if selCtx == nil || selCtx.AgenticSession == nil {
		return ""
	}
	return strings.TrimSpace(selCtx.AgenticSession.PreviousModel)
}

func selectionContextContainsModel(selCtx *selection.SelectionContext, model string) bool {
	if selCtx == nil || model == "" {
		return false
	}
	for _, candidate := range selCtx.CandidateModels {
		if candidate.Model == model || candidate.LoRAName == model {
			return true
		}
	}
	return false
}

func (r *OpenAIRouter) configuredBackendModel(model string) bool {
	if r == nil || r.Config == nil || strings.TrimSpace(model) == "" {
		return false
	}
	if _, ok := r.Config.ModelConfig[model]; ok {
		return true
	}
	return model == r.Config.DefaultModel
}

func cloneSelectionScores(scores map[string]float64) map[string]float64 {
	if scores == nil {
		return nil
	}
	clone := make(map[string]float64, len(scores))
	for key, value := range scores {
		clone[key] = value
	}
	return clone
}
