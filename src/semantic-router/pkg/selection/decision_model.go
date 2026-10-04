package selection

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var (
	ErrDecisionModelInvocation = errors.New("decision model selector invocation failed")
	ErrDecisionModelAnswer     = errors.New("decision model selector answer is invalid")
)

// DecisionModelChoice is one candidate offered to the decision model.
type DecisionModelChoice struct {
	Key         string
	Description string
}

// DecisionModelAnswer is the model's Choice answer over the candidates.
type DecisionModelAnswer struct {
	Choice        string
	Probabilities map[string]float64
	Confidence    float64
}

// DecisionModelInvoke asks the configured deployment one Choice question about
// the request; extproc binds it to the model runtime.
type DecisionModelInvoke func(
	ctx context.Context,
	instructions string,
	state string,
	choices []DecisionModelChoice,
) (DecisionModelAnswer, error)

// DecisionModelSelector asks a decision model which of the decision's
// ModelRefs should answer. The probabilities become the candidate scores;
// any failure falls back through the Router's selection fallback.
type DecisionModelSelector struct {
	config       config.DecisionSelectionConfig
	invoke       DecisionModelInvoke
	descriptions map[string]string
}

func NewDecisionModelSelector(
	cfg config.DecisionSelectionConfig,
	invoke DecisionModelInvoke,
	descriptions map[string]string,
) *DecisionModelSelector {
	return &DecisionModelSelector{config: cfg, invoke: invoke, descriptions: descriptions}
}

func (s *DecisionModelSelector) Method() SelectionMethod { return MethodDecision }

func (s *DecisionModelSelector) Tier() AlgorithmTier { return TierSupported }

func (s *DecisionModelSelector) ExternalDependencies() []Dependency {
	return []Dependency{{
		Name:        "Decision model runtime",
		Type:        DependencyExternalService,
		Description: "Requires a ready model_runtime deployment; selection falls back when it is not ready",
		Required:    false,
	}}
}

func (s *DecisionModelSelector) UpdateFeedback(context.Context, *Feedback) error { return nil }

func (s *DecisionModelSelector) Select(ctx context.Context, selCtx *SelectionContext) (*SelectionResult, error) {
	if err := ValidateSelectionContext(selCtx); err != nil {
		return nil, err
	}
	if s.invoke == nil {
		return nil, fmt.Errorf("%w: no model runtime is bound", ErrDecisionModelInvocation)
	}
	choices := s.choices(selCtx.CandidateModels)
	answer, err := s.invoke(ctx, s.config.Instructions, selCtx.Query, choices)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, fmt.Errorf("%w: %w", ErrDecisionModelInvocation, err)
	}
	selected := promptCandidateByModel(selCtx.CandidateModels, answer.Choice)
	if selected == nil {
		return nil, fmt.Errorf("%w: %q is not a candidate", ErrDecisionModelAnswer, answer.Choice)
	}
	scores := make(CandidateScores, 0, len(selCtx.CandidateModels))
	all := make(map[string]float64, len(selCtx.CandidateModels))
	for _, candidate := range selCtx.CandidateModels {
		probability, ok := answer.Probabilities[candidate.Model]
		if !ok {
			return nil, fmt.Errorf("%w: no probability for %q", ErrDecisionModelAnswer, candidate.Model)
		}
		scores = append(scores, CandidateScore{Candidate: candidate, Score: probability})
		all[candidate.Model] = probability
	}
	return &SelectionResult{
		SelectedModel:     selected.Model,
		SelectedCandidate: selected,
		LoRAName:          selected.LoRAName,
		Score:             answer.Probabilities[selected.Model],
		Confidence:        answer.Confidence,
		Method:            MethodDecision,
		Tier:              s.Tier(),
		Reasoning:         fmt.Sprintf("decision model %s chose %s", s.config.Deployment, selected.Model),
		CandidateScores:   scores,
		ScoreDirection:    HigherIsBetter,
		AllScores:         all,
	}, nil
}

func (s *DecisionModelSelector) choices(candidates []config.ModelRef) []DecisionModelChoice {
	choices := make([]DecisionModelChoice, 0, len(candidates))
	for _, candidate := range candidates {
		description := strings.TrimSpace(s.config.Candidates[candidate.Model])
		if description == "" {
			description = strings.TrimSpace(s.descriptions[candidate.Model])
		}
		if description == "" {
			description = candidate.Model
		}
		choices = append(choices, DecisionModelChoice{Key: candidate.Model, Description: description})
	}
	return choices
}
