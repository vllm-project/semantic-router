package classification

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// decisionJudgment is prepared once with a recipe's exact resource and model
// card. Requests do not discover a model, resolve another recipe or wait for
// a new generation to become ready.
type decisionJudgment struct {
	deployment string
	decider    modelservice.Decider
	card       modelservice.ModelCard
	attached   *serving.Runtime
	plan       modelservice.TaskPlan
	scan       int
	// remote is set for a systemone deployment.
	remote bool
}

func newDecisionJudgment(models *classifierModelRuntime, consumer, taskID string, question *modelservice.Question) (*decisionJudgment, error) {
	if models == nil || models.cfg == nil || models.runtime == nil {
		return nil, nil
	}
	decider, available := models.decider()
	if !available {
		return nil, nil
	}
	spec, ok, err := models.task(consumer)
	if !ok || err != nil {
		return nil, err
	}
	name, deployment := spec.Binding.Deployment, spec.Deployment
	if !deployment.IsModelRuntime() && !deployment.IsSystemOne() {
		return nil, nil
	}
	if spec.Binding.Head != "" || spec.Binding.OperatingPoint != nil {
		return nil, nil
	}
	_, explicit := models.plan.Lookup(models.recipe, consumer)
	generic := explicit && spec.Binding.Contract == config.DecisionTaskContract
	j := &decisionJudgment{deployment: name, decider: decider, scan: deployment.ScanBudget(), remote: deployment.IsSystemOne()}
	card, ready := modelservice.ModelCard{}, false
	// Explicit generic tasks have a model-independent adapter. Implicit and
	// native bindings still require metadata to select their precise adapter;
	// readiness must not silently choose between PII verdicts and token spans.
	// A systemone deployment's card is the one its API lists while the
	// generation prepares.
	if generic && !deployment.Managed() && !deployment.IsSystemOne() {
		j.attached = models.runtime
		card, ready = models.runtime.CurrentDeploymentCard(name)
	} else {
		card, err = models.runtime.DeploymentCard(context.Background(), name, deployment)
		if err != nil {
			return nil, err
		}
		ready = true
	}
	j.card = card
	if ready && !card.Serves("decisions") {
		if generic {
			return nil, fmt.Errorf("%w: deployment %q does not serve decisions required by %s", modelservice.ErrRejected, name, consumer)
		}
		return nil, nil
	}
	// A precise specialist may support only Span. Keep its existing adapter
	// unless the author explicitly selected the generic verdict contract.
	if ready && !card.Answers("noul") && (taskID == "pii_presence" || taskID == "hallucination") {
		if !generic && (card.Answers("span") || card.HasPreset("pii") || card.HasPreset("halu")) {
			return nil, nil
		}
	}
	definition, exists := modelservice.BuiltinTask(taskID)
	if !exists {
		return nil, fmt.Errorf("unknown judgment task %s", taskID)
	}
	q := definition.Question
	questionID := consumer + ":" + taskID
	if question != nil {
		q = *question
		if q.ID != "" {
			questionID = q.ID
		}
	}
	q.ID = questionID
	j.plan, err = j.preparePlan(definition, q)
	if err != nil {
		return nil, err
	}
	return j, nil
}

func (j *decisionJudgment) ask(ctx context.Context, input modelservice.Request) (modelservice.Answer, error) {
	response, err := j.askPlans(ctx, input, []modelservice.TaskPlan{j.plan})
	if err != nil {
		return modelservice.Answer{}, err
	}
	answer := response.Answers[j.plan.Question.ID]
	if answer.Error != "" {
		return answer, decisionAnswerError(answer.Error)
	}
	return answer, nil
}

// preparePlan preserves model-independent input checks while an attached
// service is offline. Its template is compiled against a ready card before
// execution; it is never sent as an uncompiled native question.
func (j *decisionJudgment) preparePlan(definition modelservice.TaskDefinition, question modelservice.Question) (modelservice.TaskPlan, error) {
	card := j.card
	if j.attached != nil {
		var ready bool
		card, ready = j.attached.CurrentDeploymentCard(j.deployment)
		if !ready {
			q, err := modelservice.PrepareTaskQuestion(definition, question)
			return modelservice.TaskPlan{Definition: definition, Question: q}, err
		}
	}
	return modelservice.CompileTask(definition, question, card)
}

func (j *decisionJudgment) askPlans(ctx context.Context, input modelservice.Request, plans []modelservice.TaskPlan) (modelservice.Response, error) {
	if j.attached != nil {
		card, ready := j.attached.CurrentDeploymentCard(j.deployment)
		if !ready {
			return modelservice.Response{}, modelservice.ErrUnavailable
		}
		compiled := make([]modelservice.TaskPlan, len(plans))
		for i, plan := range plans {
			var err error
			compiled[i], err = modelservice.CompileTask(plan.Definition, plan.Question, card)
			if err != nil {
				return modelservice.Response{}, err
			}
		}
		plans = compiled
	}
	input.MaxTokens = j.scan
	response, _, err := modelservice.ExecuteTaskPlans(ctx, j.decider, j.deployment, input, plans)
	return response, err
}

// explicitSpanBinding reports whether the configuration writes a
// token_spans.v1 contract for consumer, which demands positions.
func explicitSpanBinding(models *classifierModelRuntime, consumer string) bool {
	spec, explicit := models.plan.Lookup(models.recipe, consumer)
	return explicit && spec.Binding.Contract == config.RemoteClassifierContractTokenSpans && !spec.Binding.ContractImplied()
}

// authoredVerdictBinding reports whether the configuration writes the
// decision.v1 contract for consumer, which asks for a yes/no verdict.
func authoredVerdictBinding(models *classifierModelRuntime, consumer string) bool {
	spec, explicit := models.plan.Lookup(models.recipe, consumer)
	return explicit && spec.Binding.Contract == config.DecisionTaskContract && !spec.Binding.ContractImplied()
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
