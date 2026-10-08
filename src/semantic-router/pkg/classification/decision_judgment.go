package classification

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// decisionJudgment is prepared once with a recipe's exact resource and model
// card. Requests do not discover a model, resolve another recipe or wait for
// a new generation to become ready.
type decisionJudgment struct {
	deployment string
	decider    modelservice.Decider
	card       modelservice.ModelCard
	plan       modelservice.TaskPlan
	scan       int
}

func newDecisionJudgment(models *classifierModelRuntime, consumer, taskID string, question *modelservice.Question) (*decisionJudgment, error) {
	if models == nil || models.cfg == nil || models.runtime == nil {
		return nil, nil
	}
	decider, available := models.decider()
	if !available {
		return nil, nil
	}
	name, deployment, ok, err := models.cfg.DecisionModelDeployment()
	if spec, explicit := models.plan.Lookup(models.recipe, consumer); explicit {
		if !spec.Deployment.IsModelRuntime() || spec.Binding.Head != "" || spec.Binding.OperatingPoint != nil {
			return nil, nil
		}
		name, deployment, ok, err = spec.Binding.Deployment, spec.Deployment, true, nil
	} else if selected, resource, found, resolveErr := models.cfg.ImplicitTaskDeployment(consumer); found {
		name, deployment, ok, err = selected, resource, found, resolveErr
	}
	if !ok || err != nil {
		return nil, err
	}
	card, err := models.runtime.DeploymentCard(context.Background(), name, deployment)
	if err != nil {
		return nil, err
	}
	if !card.Serves("decisions") {
		return nil, nil
	}
	// A precise specialist may support only Span. Keep its existing adapter
	// unless the author explicitly selected the generic verdict contract.
	if !card.Answers("noul") && (taskID == "pii_presence" || taskID == "hallucination") {
		spec, explicit := models.plan.Lookup(models.recipe, consumer)
		if (!explicit || spec.Binding.Contract != config.DecisionTaskContract) && (card.Answers("span") || card.HasPreset("pii") || card.HasPreset("halu")) {
			return nil, nil
		}
	}
	definition, exists := modelservice.BuiltinTask(taskID)
	if !exists {
		return nil, fmt.Errorf("unknown judgment task %s", taskID)
	}
	q := definition.Question
	if question != nil {
		q = *question
	}
	q.ID = consumer + ":" + taskID
	plan, err := modelservice.CompileTask(definition, q, card)
	if err != nil {
		return nil, err
	}
	return &decisionJudgment{deployment: name, decider: decider, card: card, plan: plan, scan: deployment.ScanBudget()}, nil
}

func (j *decisionJudgment) ask(ctx context.Context, input modelservice.Request) (modelservice.Answer, error) {
	input.MaxTokens = j.scan
	response, _, err := modelservice.ExecuteTaskPlans(ctx, j.decider, j.deployment, input, []modelservice.TaskPlan{j.plan})
	if err != nil {
		return modelservice.Answer{}, err
	}
	answer := response.Answers[j.plan.Question.ID]
	if answer.Error != "" {
		return answer, decisionAnswerError(answer.Error)
	}
	return answer, nil
}

func explicitSpanBinding(models *classifierModelRuntime, consumer string) bool {
	spec, explicit := models.plan.Lookup(models.recipe, consumer)
	return explicit && spec.Binding.Contract == config.RemoteClassifierContractTokenSpans
}

func decisionAnswerError(code string) error {
	switch code {
	case "max_length_exceeded", "input_limit", "input_too_long":
		return fmt.Errorf("%w: %s", binding.ErrInputLimit, code)
	case "scan_budget_exceeded":
		return fmt.Errorf("%w: %s", binding.ErrScanBudget, code)
	case "deadline_exceeded":
		return context.DeadlineExceeded
	case "unavailable", "not_ready":
		return modelservice.ErrUnavailable
	default:
		return fmt.Errorf("%w: %s", binding.ErrInvalidResult, code)
	}
}
