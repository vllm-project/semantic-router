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
	return s.result.WithScores(s.result.ScoresFor(selCtx.CandidateModels)), nil
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
