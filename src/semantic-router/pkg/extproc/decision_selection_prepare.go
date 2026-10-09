package extproc

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// prepareDecisionSelectionCards admits selectors while building a generation.
// Requests only read this snapshot; no discovery runs on their selection path.
func prepareDecisionSelectionCards(cfg *config.RouterConfig, provider interface {
	Card(context.Context, string) (modelservice.ModelCard, error)
},
) (map[string]modelservice.ModelCard, error) {
	cards := map[string]modelservice.ModelCard{}
	if cfg == nil {
		return cards, nil
	}
	definition, _ := modelservice.BuiltinTask("model_selection")
	for _, recipe := range cfg.ReachableRoutingRecipes() {
		for _, decision := range recipe.Profile.Decisions {
			if decision.Algorithm == nil || decision.Algorithm.Type != config.DecisionAlgorithmDecision || decision.Algorithm.Decision == nil {
				continue
			}
			selector := *decision.Algorithm.Decision
			deployment := cfg.DecisionSelectorDeployment(selector)
			card, exists := cards[deployment]
			if !exists {
				if provider == nil {
					return nil, fmt.Errorf("decision selector %s requires a model runtime", decision.Name)
				}
				ctx, cancel := context.WithTimeout(context.Background(), selector.EffectiveTimeout())
				var err error
				card, err = provider.Card(ctx, deployment)
				cancel()
				if err != nil {
					return nil, fmt.Errorf("prepare decision selector %s: %w", decision.Name, err)
				}
				if !card.Serves("decisions") {
					return nil, fmt.Errorf("decision selector %s requires the decisions surface", decision.Name)
				}
				cards[deployment] = card
			}
			question := modelservice.Question{ID: decisionSelectorQuestionID, Type: config.DecisionQuestionChoice, Instructions: selector.Instructions, Truncate: true}
			for _, candidate := range decision.ModelRefs {
				question.Choices = append(question.Choices, modelservice.Choice{Key: candidate.Model, Description: selector.Candidates[candidate.Model]})
			}
			if _, err := modelservice.CompileTask(definition, question, card); err != nil {
				return nil, fmt.Errorf("prepare decision selector %s: %w", decision.Name, err)
			}
		}
	}
	return cards, nil
}
