package extproc

import (
	"context"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

const decisionSelectorQuestionID = "selector"

// newDecisionModelSelector binds the decision selector to the model runtime,
// or to a systemone deployment's remote API. A selector that names no
// deployment asks the decision model's deployment.
func (r *OpenAIRouter) newDecisionModelSelector(cfg config.SystemOneSelectionConfig) selection.Selector {
	descriptions := make(map[string]string, len(r.Config.ModelConfig))
	for model, params := range r.Config.ModelConfig {
		descriptions[model] = params.Description
	}
	cfg.Deployment = r.Config.SystemOneSelectorDeployment(cfg)
	timeout := r.systemOneCallTimeout(cfg.Deployment, cfg.TimeoutMs, cfg.EffectiveTimeout())
	invoke := func(
		ctx context.Context,
		instructions string,
		state string,
		choices []selection.DecisionModelChoice,
	) (selection.DecisionModelAnswer, error) {
		callCtx, cancel := context.WithTimeout(ctx, timeout)
		defer cancel()
		// Selection routes, so a long state is read only as far as its first tokens.
		question := modelservice.Question{ID: decisionSelectorQuestionID, Type: config.SystemOneQuestionChoice, Instructions: instructions, Truncate: true}
		for _, choice := range choices {
			question.Choices = append(question.Choices, modelservice.Choice{Key: choice.Key, Description: choice.Description})
		}
		definition, _ := modelservice.BuiltinTask("model_selection")
		answer, err := r.askSystemOneTask(callCtx, cfg.Deployment, definition, question, modelservice.Request{State: state})
		if err != nil {
			return selection.DecisionModelAnswer{}, err
		}
		return selection.DecisionModelAnswer{Choice: answer.Choice, Probabilities: answer.Probabilities, Confidence: answer.Confidence}, nil
	}
	return selection.NewDecisionModelSelector(cfg, invoke, descriptions)
}

// systemOneCallTimeout is the deadline of a call a selector or an
// escalation's check makes: its own timeout_ms, else a systemone
// deployment's timeout, else the System One default.
func (r *OpenAIRouter) systemOneCallTimeout(deployment string, timeoutMs int, effective time.Duration) time.Duration {
	if remote := r.Config.ModelDeployments[deployment]; remote.IsSystemOne() && timeoutMs == 0 && remote.TimeoutMs > 0 {
		return remote.CallTimeout()
	}
	return effective
}

// askSystemOneTask asks a built-in task's question on a deployment: the
// model runtime with its card prepared with the generation, or a systemone
// deployment's API, whose listing may have answered since preparation. The
// question goes out in the form the card's family was trained on, and its
// answer comes back under the question's ID.
func (r *OpenAIRouter) askSystemOneTask(
	ctx context.Context,
	deployment string,
	definition modelservice.TaskDefinition,
	question modelservice.Question,
	request modelservice.Request,
) (modelservice.Answer, error) {
	card, ready := r.decisionCards[deployment]
	asked := r.decisionDecider
	if asked == nil {
		asked = modelservice.Default()
	}
	if r.Config.ModelDeployments[deployment].IsSystemOne() {
		if r.systemOneDecider == nil {
			return modelservice.Answer{}, modelservice.ErrUnavailable
		}
		card, ready = r.systemOneDecider.Card(deployment)
		asked = r.systemOneDecider
	}
	if !ready {
		return modelservice.Answer{}, fmt.Errorf("%s requires a prepared model card for deployment %q", definition.ID, deployment)
	}
	plan, err := modelservice.CompileTask(definition, question, card)
	if err != nil {
		return modelservice.Answer{}, err
	}
	request.MaxTokens = r.Config.ModelDeployments[deployment].ScanBudget()
	response, _, err := modelservice.ExecuteTaskPlans(ctx, asked, deployment, request, []modelservice.TaskPlan{plan})
	if err != nil {
		modelservice.RecordUnknown(deployment, modelservice.ErrorReason(err), 1)
		return modelservice.Answer{}, err
	}
	answer, ok := response.Answers[question.ID]
	if !ok || answer.Error != "" {
		modelservice.RecordUnknown(deployment, "answer_error", 1)
		return modelservice.Answer{}, fmt.Errorf("%w: %s", selection.ErrDecisionModelAnswer, answer.Error)
	}
	return answer, nil
}
