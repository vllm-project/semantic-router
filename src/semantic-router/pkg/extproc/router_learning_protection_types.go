package extproc

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

type learningSelectionResult struct {
	result *selection.SelectionResult
}

func (s learningSelectionResult) Select(_ context.Context, selCtx *selection.SelectionContext) (*selection.SelectionResult, error) {
	if s.result == nil {
		return nil, selection.ErrSelectionResultRequired
	}
	result := *s.result
	result.AllScores = cloneSelectionScores(s.result.AllScores)
	if result.AllScores == nil {
		result.AllScores = make(map[string]float64, len(selCtx.CandidateModels))
	}
	for _, candidate := range selCtx.CandidateModels {
		if _, ok := result.AllScores[candidate.Model]; !ok {
			result.AllScores[candidate.Model] = 0
		}
	}
	if _, ok := result.AllScores[result.SelectedModel]; !ok {
		result.AllScores[result.SelectedModel] = result.Score
	}
	return &result, nil
}

func (s learningSelectionResult) Method() selection.SelectionMethod {
	if s.result == nil || s.result.Method == "" {
		return selection.MethodStatic
	}
	return s.result.Method
}

func (s learningSelectionResult) UpdateFeedback(context.Context, *selection.Feedback) error {
	return nil
}
func (s learningSelectionResult) Tier() selection.AlgorithmTier                { return selection.TierSupported }
func (s learningSelectionResult) ExternalDependencies() []selection.Dependency { return nil }
