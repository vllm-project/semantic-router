package extproc

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

const decisionSelectorQuestionID = "selector"

// newDecisionModelSelector binds the decision selector to the model runtime,
// or to a systemone deployment's remote API. A selector that names no
// deployment asks the decision model's deployment.
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
	deployment := r.Config.ModelDeployments[cfg.Deployment]
	timeout := cfg.EffectiveTimeout()
	if deployment.IsSystemOne() && cfg.TimeoutMs == 0 && deployment.TimeoutMs > 0 {
		timeout = deployment.CallTimeout()
	}
	scan := deployment.ScanBudget()
	prepared, preparedOK := r.decisionCards[cfg.Deployment]
	invoke := func(
		ctx context.Context,
		instructions string,
		state string,
		choices []selection.DecisionModelChoice,
	) (selection.DecisionModelAnswer, error) {
		callCtx, cancel := context.WithTimeout(ctx, timeout)
		defer cancel()
		card, ready, asked := prepared, preparedOK, decider
		if deployment.IsSystemOne() {
			if r.systemOneDecider == nil {
				return selection.DecisionModelAnswer{}, modelservice.ErrUnavailable
			}
			// The API's listing may have answered since preparation.
			card, ready = r.systemOneDecider.Card(cfg.Deployment)
			asked = r.systemOneDecider
		}
		if !ready {
			return selection.DecisionModelAnswer{}, fmt.Errorf("decision selector requires a prepared model card")
		}
		// Selection routes, so a long state is read only as far as its first tokens.
		question := modelservice.Question{ID: decisionSelectorQuestionID, Type: config.DecisionQuestionChoice, Instructions: instructions, Truncate: true}
		for _, choice := range choices {
			question.Choices = append(question.Choices, modelservice.Choice{Key: choice.Key, Description: choice.Description})
		}
		definition, _ := modelservice.BuiltinTask("model_selection")
		plan, err := modelservice.CompileTask(definition, question, card)
		if err != nil {
			return selection.DecisionModelAnswer{}, err
		}
		response, _, err := modelservice.ExecuteTaskPlans(callCtx, asked, cfg.Deployment, modelservice.Request{State: state, MaxTokens: scan}, []modelservice.TaskPlan{plan})
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
