package extproc

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

const decisionSelectorQuestionID = "selector"

// newDecisionModelSelector binds the decision selector to the model runtime.
// A selector that names no deployment asks the decision model's deployment.
func (r *OpenAIRouter) newDecisionModelSelector(cfg config.DecisionSelectionConfig) selection.Selector {
	descriptions := make(map[string]string, len(r.Config.ModelConfig))
	for model, params := range r.Config.ModelConfig {
		descriptions[model] = params.Description
	}
	decider := r.decisionDecider
	if decider == nil {
		decider = modelservice.Default()
	}
	cfg.Deployment = r.Config.DecisionSelectorDeployment(cfg)
	scan := r.Config.ModelDeployments[cfg.Deployment].ScanBudget()
	invoke := func(
		ctx context.Context,
		instructions string,
		state string,
		choices []selection.DecisionModelChoice,
	) (selection.DecisionModelAnswer, error) {
		callCtx, cancel := context.WithTimeout(ctx, cfg.EffectiveTimeout())
		defer cancel()
		// Selection routes, so a long state is read only as far as its first tokens.
		question := modelservice.Question{ID: decisionSelectorQuestionID, Type: config.DecisionQuestionChoice, Instructions: instructions, Truncate: true}
		for _, choice := range choices {
			question.Choices = append(question.Choices, modelservice.Choice{Key: choice.Key, Description: choice.Description})
		}
		response, err := decider.Decide(callCtx, cfg.Deployment, modelservice.Request{State: state, Questions: []modelservice.Question{question}, MaxTokens: scan})
		if err != nil {
			modelservice.RecordUnknown(cfg.Deployment, modelservice.ErrorReason(err), 1)
			return selection.DecisionModelAnswer{}, err
		}
		answer, ok := response.Answers[decisionSelectorQuestionID]
		if !ok || answer.Error != "" {
			modelservice.RecordUnknown(cfg.Deployment, "answer_error", 1)
			return selection.DecisionModelAnswer{}, fmt.Errorf("%w: %s", selection.ErrDecisionModelAnswer, answer.Error)
		}
		return selection.DecisionModelAnswer{Choice: answer.Choice, Probabilities: answer.Probabilities, Confidence: answer.Confidence}, nil
	}
	return selection.NewDecisionModelSelector(cfg, invoke, descriptions)
}
