package match

import (
	"context"
	"fmt"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

// The state parts a judge reads: the request, the earlier user turns when
// the feature reads them, and a candidate, which the model reads as context.
const (
	requestPart   = "request"
	historyPart   = "history"
	candidatePart = "candidate"
)

// questionID is the ID a judge's question carries on the wire: the Router's
// own, with a ':', so it never turns on an adapter trained under another ID.
func questionID(name string) string {
	return "match:" + name
}

// judge scores the candidates with the judge's question and keeps them by
// its rules, best first.
func (m *Matcher) judge(ctx context.Context, spec config.MatchJudge, input Input, candidates []Scored) ([]Scored, Stage, error) {
	stage := Stage{Name: StageJudge, Method: spec.Preset, Candidates: len(candidates)}
	if spec.Question != nil {
		stage.Method = "question"
	}
	if len(candidates) == 0 {
		stage.Outcome = OutcomeEmpty
		return nil, stage, nil
	}
	stage.Deployment = m.cfg.MatchJudgeDeployment(spec)
	started := time.Now()
	scores, asked, err := m.score(ctx, spec, stage.Deployment, input, candidates)
	stage.Duration = time.Since(started)
	if err != nil {
		stage.Reason = reason(err)
		modelservice.RecordUnknown(stage.Deployment, stage.Reason, asked)
		return nil, stage, err
	}
	kept := keep(spec, candidates, scores)
	stage.Kept, stage.Outcome = len(kept), OutcomeOK
	return kept, stage, nil
}

// score asks the judge's question, compiled for the deployment's model, and
// returns each candidate's score and how many questions it asked. A choice
// question chooses among the candidates about the request, at most 255 of
// them; any other is asked about each candidate, each a state of the same
// call, and a candidate without text scores 0 unasked.
func (m *Matcher) score(ctx context.Context, spec config.MatchJudge, deployment string, input Input, candidates []Scored) ([]float64, int, error) {
	chooses := spec.Chooses()
	if chooses && len(candidates) == 1 {
		return []float64{1}, 0, nil
	}
	card, ok := m.answerer.Card(deployment)
	if !ok {
		return nil, 0, fmt.Errorf("%w: deployment %q has no model card ready", modelservice.ErrUnavailable, deployment)
	}
	definition, question := judgeQuestion(spec)
	if chooses {
		for _, candidate := range candidates {
			question.Choices = append(question.Choices, modelservice.Choice{Key: candidate.ID, Description: candidate.Text})
		}
	}
	plan, err := modelservice.CompileTask(definition, question, card)
	if err != nil {
		return nil, 0, err
	}
	ctx, cancel := context.WithTimeout(ctx, m.timeout(spec, deployment))
	defer cancel()
	scores := make([]float64, len(candidates))
	if chooses {
		answer, err := m.ask(ctx, deployment, input.State(), plan)
		if err != nil {
			return nil, 1, err
		}
		for index, candidate := range candidates {
			scores[index] = answer.Probabilities[candidate.ID]
		}
		return scores, 1, nil
	}
	var asked []int
	for index, candidate := range candidates {
		if strings.TrimSpace(candidate.Text) != "" {
			asked = append(asked, index)
		}
	}
	errs := make([]error, len(asked))
	fan(ctx, len(asked), func(ctx context.Context, i int) {
		parts := map[string]string{requestPart: input.Request, candidatePart: candidates[asked[i]].Text}
		if strings.TrimSpace(input.History) != "" {
			parts[historyPart] = input.History
		}
		state := modelservice.Request{Parts: parts}
		answer, err := m.ask(ctx, deployment, state, plan)
		if err != nil {
			errs[i] = err
			return
		}
		scores[asked[i]] = answer.Noul
		if question.Type == config.SystemOneQuestionScore {
			scores[asked[i]] = answer.Score
		}
	})
	for _, err := range errs {
		if err != nil {
			return nil, len(asked), err
		}
	}
	return scores, len(asked), nil
}

// Prepare compiles a judge's question for the card of the deployment that
// answers it, as a request does, so a generation refuses a judge its
// deployment cannot answer before a request asks it.
func Prepare(spec config.MatchJudge, card modelservice.ModelCard) error {
	definition, question := judgeQuestion(spec)
	_, err := modelservice.CompileTask(definition, question, card)
	return err
}

// judgeQuestion is the task a judge's question compiles as and the question
// under its own ID: a preset's canonical question, or the custom one. Like
// the presets', a custom question reads a long candidate's first tokens.
func judgeQuestion(spec config.MatchJudge) (modelservice.TaskDefinition, modelservice.Question) {
	if custom := spec.Question; custom != nil {
		question := modelservice.Question{ID: questionID("question"), Type: custom.Type, Instructions: custom.Instructions, Levels: slices.Clone(custom.Levels), Truncate: true}
		for _, choice := range custom.Choices {
			question.Choices = append(question.Choices, modelservice.Choice{Key: choice.Key, Description: choice.Description})
		}
		return modelservice.TaskDefinition{ID: question.ID, Stage: presets.StageRequest, Output: question.Type}, question
	}
	preset, _ := presets.Lookup(spec.Preset, presets.StageRequest)
	question := modelservice.PresetQuestion(questionID(preset.ID), preset)
	return modelservice.TaskDefinition{ID: question.ID, Stage: preset.Stage, Output: preset.Type, Preset: preset.ID}, question
}

// ask asks one question about one state; an answer the model could not give
// fails the judge.
func (m *Matcher) ask(ctx context.Context, deployment string, state modelservice.Request, plan modelservice.TaskPlan) (modelservice.Answer, error) {
	response, _, err := modelservice.ExecuteTaskPlans(ctx, m.answerer, deployment, state, []modelservice.TaskPlan{plan})
	if err != nil {
		return modelservice.Answer{}, err
	}
	answer, ok := response.Answers[plan.Question.ID]
	switch {
	case !ok:
		return modelservice.Answer{}, answerFailure("missing_answer")
	case answer.Error != "":
		return modelservice.Answer{}, answerFailure(answer.Error)
	}
	return answer, nil
}

// answerFailure is an answer the model could not give, by its error code.
type answerFailure string

func (f answerFailure) Error() string {
	return "match: the judge's answer failed: " + string(f)
}

// timeout is the judge's deadline: its own, else a systemone deployment's
// own, else the default.
func (m *Matcher) timeout(spec config.MatchJudge, deployment string) time.Duration {
	if remote := m.cfg.ModelDeployments[deployment]; spec.TimeoutMs == 0 && remote.IsSystemOne() && remote.TimeoutMs > 0 {
		return remote.CallTimeout()
	}
	return spec.EffectiveTimeout()
}

// keep orders the judged candidates by score, ties in the order they came,
// and keeps those the judge's rules keep: at least min_score, at least
// min_ratio of the best score, and at most top_k.
func keep(spec config.MatchJudge, candidates []Scored, scores []float64) []Scored {
	judged := make([]Scored, len(candidates))
	best := 0.0
	for index, candidate := range candidates {
		candidate.Score = scores[index]
		judged[index], best = candidate, max(best, candidate.Score)
	}
	sort.SliceStable(judged, func(i, j int) bool { return judged[i].Score > judged[j].Score })
	kept := judged[:0]
	for _, candidate := range judged {
		if (spec.MinScore != nil && candidate.Score < *spec.MinScore) || (spec.MinRatio != nil && candidate.Score < *spec.MinRatio*best) {
			continue
		}
		kept = append(kept, candidate)
	}
	if spec.TopK > 0 && len(kept) > spec.TopK {
		kept = kept[:spec.TopK]
	}
	return kept
}

// fan runs work for each of n states as participants of the context's
// bundle, so that one call carries them all; a caller without a bundle gets
// one of its own.
func fan(ctx context.Context, n int, work func(ctx context.Context, i int)) {
	if modelservice.InBundle(ctx) {
		modelservice.Fan(ctx, n, func(i int) { work(ctx, i) })
		return
	}
	stage, bundle := modelservice.WithBundle(ctx, 0)
	leave := bundle.Join()
	defer leave()
	modelservice.Fan(stage, n, func(i int) { work(stage, i) })
}
