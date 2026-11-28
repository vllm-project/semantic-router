package extproc

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selectiontrace"
)

// The IDs a candidate estimate's questions carry on the wire: the Router's
// own, with a ':', so they never turn on an adapter trained under another ID.
const (
	candidateSuccessQuestionID = "estimate:" + presets.CandidateSuccess
	outputLengthQuestionID     = "estimate:" + presets.OutputLength
)

// The state parts an estimate reads: the request, and the candidate's
// description, which the model reads as context.
const (
	estimateRequestPart   = "request"
	estimateCandidatePart = "candidate"
)

// multiFactorEstimates binds a multi_factor decision's candidate estimates to
// the deployment that answers them.
func (r *OpenAIRouter) multiFactorEstimates(cfg config.MultiFactorEstimatesConfig) *selection.MultiFactorEstimates {
	deployment := r.Config.MultiFactorEstimatesDeployment(cfg)
	timeout := r.systemOneCallTimeout(deployment, cfg.TimeoutMs, cfg.EffectiveTimeout())
	return &selection.MultiFactorEstimates{
		Mode:       cfg.Mode,
		Deployment: deployment,
		Estimate: func(ctx context.Context, request string, candidates []config.ModelRef) []selection.CandidateAnswer {
			return r.estimateCandidates(ctx, deployment, timeout, request, candidates)
		},
	}
}

// compileCandidateEstimates compiles an estimate's two questions, the
// candidate_success and output_length presets, for a deployment's card.
func compileCandidateEstimates(card modelservice.ModelCard) ([]modelservice.TaskPlan, error) {
	plans := make([]modelservice.TaskPlan, 0, 2)
	for _, task := range []struct{ id, preset string }{
		{candidateSuccessQuestionID, presets.CandidateSuccess},
		{outputLengthQuestionID, presets.OutputLength},
	} {
		preset, _ := presets.Lookup(task.preset, presets.StageRequest)
		question := modelservice.PresetQuestion(task.id, preset)
		definition := modelservice.TaskDefinition{ID: task.id, Stage: preset.Stage, Output: preset.Type, Preset: preset.ID}
		plan, err := modelservice.CompileTask(definition, question, card)
		if err != nil {
			return nil, err
		}
		plans = append(plans, plan)
	}
	return plans, nil
}

// estimateCandidates asks the deployment both questions about each
// candidate, each candidate its own state, as participants of one bundle,
// which sends them in one call. The timeout bounds the whole call; a
// candidate whose answer fails or comes late is marked so, and the request
// goes on.
func (r *OpenAIRouter) estimateCandidates(
	ctx context.Context,
	deployment string,
	timeout time.Duration,
	request string,
	candidates []config.ModelRef,
) []selection.CandidateAnswer {
	answers := make([]selection.CandidateAnswer, len(candidates))
	answerer := routerMatchAnswerer{r}
	card, ready := answerer.Card(deployment)
	if !ready {
		return unavailableAnswers(answers, "not_ready")
	}
	plans, err := compileCandidateEstimates(card)
	if err != nil {
		return unavailableAnswers(answers, modelservice.ErrorReason(err))
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	modelservice.FanStates(ctx, len(candidates), func(ctx context.Context, i int) {
		state := modelservice.Request{Parts: map[string]string{
			estimateRequestPart:   request,
			estimateCandidatePart: r.estimateCandidateDescription(candidates[i]),
		}}
		response, _, err := modelservice.ExecuteTaskPlans(ctx, answerer, deployment, state, plans)
		answers[i] = candidateAnswer(response, err)
		if answers[i].Status != selectiontrace.EstimateAvailable {
			modelservice.RecordUnknown(deployment, answers[i].Reason, len(plans))
		}
	})
	return answers
}

// candidateAnswer reads one state's answers: P(yes) of candidate_success and
// the distribution over output_length's bands.
func candidateAnswer(response modelservice.Response, err error) selection.CandidateAnswer {
	if err != nil {
		status := selectiontrace.EstimateUnavailable
		if errors.Is(err, context.DeadlineExceeded) {
			status = selectiontrace.EstimateDeadlineExceeded
		}
		return selection.CandidateAnswer{Status: status, Reason: modelservice.ErrorReason(err)}
	}
	success, ok := response.Answers[candidateSuccessQuestionID]
	if !ok {
		return selection.CandidateAnswer{Status: selectiontrace.EstimateUnavailable, Reason: "missing_answer"}
	}
	length, ok := response.Answers[outputLengthQuestionID]
	if !ok {
		return selection.CandidateAnswer{Status: selectiontrace.EstimateUnavailable, Reason: "missing_answer"}
	}
	for _, answer := range []modelservice.Answer{success, length} {
		if answer.Error == "deadline_exceeded" {
			return selection.CandidateAnswer{Status: selectiontrace.EstimateDeadlineExceeded, Reason: answer.Error}
		}
		if answer.Error != "" {
			return selection.CandidateAnswer{Status: selectiontrace.EstimateUnavailable, Reason: answer.Error}
		}
	}
	return selection.CandidateAnswer{
		Status:             selectiontrace.EstimateAvailable,
		SuccessProbability: success.Noul,
		OutputLength:       length.Probabilities,
	}
}

func unavailableAnswers(answers []selection.CandidateAnswer, reason string) []selection.CandidateAnswer {
	for i := range answers {
		answers[i] = selection.CandidateAnswer{Status: selectiontrace.EstimateUnavailable, Reason: reason}
	}
	return answers
}

// estimateCandidateDescription is the candidate part of an estimate's state:
// the model and its description, then its adapter and reasoning setting, so
// that two efforts of one model are two candidates to the estimating model.
// The ref's reasoning_description, when set, follows.
func (r *OpenAIRouter) estimateCandidateDescription(ref config.ModelRef) string {
	lines := []string{ref.Model}
	if description := strings.TrimSpace(r.Config.ModelConfig[ref.Model].Description); description != "" {
		lines[0] += ": " + description
	}
	if ref.LoRAName != "" {
		lines = append(lines, "Adapter: "+ref.LoRAName)
	}
	switch {
	case ref.UseReasoning != nil && !*ref.UseReasoning:
		lines = append(lines, "Reasoning: off")
	case ref.ReasoningEffort != "":
		lines = append(lines, "Reasoning: on, effort "+ref.ReasoningEffort)
	case ref.UseReasoning != nil:
		lines = append(lines, "Reasoning: on")
	}
	if description := strings.TrimSpace(ref.ReasoningDescription); description != "" {
		lines = append(lines, description)
	}
	return strings.Join(lines, "\n")
}
