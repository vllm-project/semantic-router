package extproc

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

// escalateAnswerChecks asks a System One deployment about an escalation's
// answers: the answer_check preset and the refusal preset, each about the
// request and one answer, at the response stage.
type escalateAnswerChecks struct {
	router     *OpenAIRouter
	deployment string
	timeout    time.Duration
}

// escalateAnswerChecker binds a Chat escalation's answer checks to their
// deployment; nil when its accept check asks no System One question.
func (r *OpenAIRouter) escalateAnswerChecker(algorithm *config.AlgorithmConfig) looper.AnswerChecker {
	if !algorithm.EscalateChecksAnswers() {
		return nil
	}
	accept := algorithm.Escalate.Accept
	deployment := r.Config.EscalateCheckDeployment(accept)
	check := config.EscalateAnswerCheck{}
	if accept.SystemOne != nil {
		check = *accept.SystemOne
	}
	timeout := r.systemOneCallTimeout(deployment, check.TimeoutMs, check.EffectiveTimeout())
	return &escalateAnswerChecks{router: r, deployment: deployment, timeout: timeout}
}

// CheckAnswer returns the probability that answer correctly and completely
// answers request.
func (c *escalateAnswerChecks) CheckAnswer(ctx context.Context, request, answer string) (float64, error) {
	reply, err := c.ask(ctx, presets.AnswerCheck, request, answer)
	if err != nil {
		return 0, err
	}
	return reply.Noul, nil
}

// Refusal returns the probability that answer refuses request, and the
// threshold at which the refusal preset flags it: the neutral one, as the
// preset has no calibrated threshold.
func (c *escalateAnswerChecks) Refusal(ctx context.Context, request, answer string) (float64, float64, error) {
	reply, err := c.ask(ctx, presets.Refusal, request, answer)
	if err != nil {
		return 0, 0, err
	}
	return reply.Probabilities["refusal"], presets.NeutralThreshold, nil
}

func (c *escalateAnswerChecks) ask(ctx context.Context, task, request, answer string) (modelservice.Answer, error) {
	definition, _ := modelservice.BuiltinTask(task)
	callCtx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	return c.router.askSystemOneTask(callCtx, c.deployment, definition, definition.Question,
		modelservice.Request{Parts: map[string]string{"request": request, "answer": answer}})
}
