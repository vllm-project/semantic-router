package selection

import (
	"context"
	"math/rand/v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// RandomSelector picks uniformly from the candidates it is given. Eligibility,
// fallback and recovery belong to the shared pipeline, never here (#3273).
// Unrelated to RLDrivenSelector.selectRandomModel, which is RL exploration.
type RandomSelector struct {
	// Overridden only by tests in this package; production has no seed knob.
	intn func(int) int
}

// NewRandomSelector creates a new Random selector
func NewRandomSelector() *RandomSelector {
	return &RandomSelector{intn: rand.IntN}
}

// Method returns the selection method type
func (s *RandomSelector) Method() SelectionMethod { return MethodRandom }

// Tier returns the production readiness tier
func (s *RandomSelector) Tier() AlgorithmTier { return TierSupported }

// ExternalDependencies returns external dependencies (none for random)
func (s *RandomSelector) ExternalDependencies() []Dependency { return []Dependency{} }

// UpdateFeedback does nothing for the random selector (no learning)
func (s *RandomSelector) UpdateFeedback(context.Context, *Feedback) error { return nil }

// Select draws one candidate uniformly from the eligible candidate models
func (s *RandomSelector) Select(_ context.Context, selCtx *SelectionContext) (*SelectionResult, error) {
	if err := ValidateSelectionContext(selCtx); err != nil {
		return nil, err
	}

	candidates := selCtx.CandidateModels
	chosen := &candidates[s.intn(len(candidates))]

	score := 1.0 / float64(len(candidates))
	allScores := make(map[string]float64, len(candidates))
	for _, candidate := range candidates {
		allScores[candidate.Model] = score
	}

	logging.Infof("[RandomSelector] %v → %s (1/%d)",
		getModelNames(candidates), chosen.Model, len(candidates))

	return &SelectionResult{
		SelectedModel: chosen.Model,
		LoRAName:      chosen.LoRAName,
		Score:         score,
		Confidence:    score,
		Method:        MethodRandom,
		Reasoning:     "uniform random selection among eligible candidates",
		AllScores:     allScores,
	}, nil
}
