package systemone

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// ErrUnresolved means no complete response satisfied the declared quality
// conditions within the available stages and request budget.
var ErrUnresolved = errors.New("native inference quality unresolved")

// Router executes an API-scoped recipe within its retained generation.
type Router interface {
	RouteSystemOne(context.Context, string, json.RawMessage, Invoke) (int, []byte, error)
}

// Candidate retains the original native response, not a reconstructed answer.
type Candidate struct {
	Stage        string
	Model        string
	Body         json.RawMessage
	Observations []Observation
	// Invocation is the actual stage response. For a judge it differs from
	// Body, which remains the selected native model's intact response.
	Invocation json.RawMessage
	History    []StageEvidence
}

// StageEvidence records the conditional arrival path inside one request. It
// is retained for evaluation, never emitted as public response metadata.
type StageEvidence struct {
	Stage, Model, Outcome string
	Response              json.RawMessage
}

// QualityEvaluator checks evidence for this exact stage and native request.
// Its implementation must distinguish unavailable evidence from a passing risk.
type QualityEvaluator func(*NativeRequest, Candidate) (bool, error)

// Executor is immutable after generation preparation. Per-request evidence,
// attempted stages and model responses never live on this shared object.
type Executor struct {
	algorithm *config.AlgorithmConfig
	policy    *LearnedPolicy
	quality   QualityEvaluator
}

func NewExecutor(algorithm *config.AlgorithmConfig, quality QualityEvaluator) (*Executor, error) {
	if algorithm == nil || !algorithm.IsNative() || algorithm.Quality == nil || len(algorithm.Stages) == 0 {
		return nil, errors.New("invalid native execution plan")
	}
	e := &Executor{algorithm: algorithm, quality: quality}
	if algorithm.Type == config.DecisionAlgorithmPolicy {
		var err error
		e.policy, err = LoadPolicy(algorithm.Policy, algorithm.Stages)
		if err != nil {
			return nil, err
		}
	}
	if algorithm.Quality.Type == "calibrated" && quality == nil {
		return nil, errors.New("calibrated native execution requires applicable evaluation evidence")
	}
	return e, nil
}

// Execute keeps the entire question bundle together. Model errors may advance
// to another declared stage, but cancellation and the physical call budget
// remain shared by every attempt through invoke.
func (e *Executor) Execute(ctx context.Context, request *NativeRequest, invoke Invoke) (result Candidate, resultErr error) {
	started := time.Now()
	defer func() {
		outcome := "resolved"
		if resultErr != nil {
			outcome = "unresolved"
			if ctx.Err() != nil {
				outcome = "canceled"
			}
		}
		metrics.RecordSystemOneRequest(e.algorithm.Type, outcome, time.Since(started).Seconds())
	}()
	stages := e.algorithm.Stages
	attempted := make(map[string]bool, len(stages))
	var candidates []Candidate
	var history []StageEvidence
	var accepted *Candidate
	for index := 0; index >= 0 && index < len(stages); {
		if err := ctx.Err(); err != nil {
			return Candidate{}, err
		}
		stage := stages[index]
		if !stage.IsEnabled() {
			index++
			continue
		}
		if attempted[stage.Name] {
			return Candidate{}, errors.New("native stage loop rejected")
		}
		attempted[stage.Name] = true
		stageStarted := time.Now()
		candidate, err := e.invokeStage(ctx, request, stage, candidates, invoke)
		if err == nil && e.policy != nil && stage.Kind == "native" && !e.policy.matches(stage, candidate) {
			metrics.RecordSystemOneStage(e.algorithm.Type, stage.Name, stage.Model, "identity_mismatch", time.Since(stageStarted).Seconds())
			return Candidate{}, errors.New("native policy response identity does not match its fitted action")
		}
		evidence := StageEvidence{Stage: stage.Name, Model: stage.Model, Response: candidate.Invocation, Outcome: "response"}
		if err != nil {
			evidence.Outcome = "error"
		}
		history = append(history, evidence)
		candidate.History = append([]StageEvidence(nil), history...)
		if err != nil {
			metrics.RecordSystemOneStage(e.algorithm.Type, stage.Name, stage.Model, "error", time.Since(stageStarted).Seconds())
			if ctx.Err() != nil {
				return Candidate{}, ctx.Err()
			}
			// Failed calls use the versioned missing-observation features.
			// Only an action present in the fitted policy may be selected;
			// the operator's optional judge remains a terminal fallback.
			if e.policy != nil {
				features := request.Features(request.Observe(nil))
				index = e.policy.Next(stage.Name, features, stages, attempted, e.algorithm.Policy.CostWeight)
				if index < 0 {
					index = terminalJudge(stages, attempted, accepted, candidates)
				}
				continue
			}
			index++
			continue
		}
		candidate.Observations = request.Observe(candidate.Body)
		if complete(candidate.Observations) {
			candidates = append(candidates, candidate)
		}
		passes, qualityErr := e.accept(request, stage, candidate)
		outcome := "rejected"
		if qualityErr != nil {
			outcome = "evaluation_error"
		} else if passes {
			outcome = "accepted"
		} else if !complete(candidate.Observations) {
			outcome = "invalid"
		}
		metrics.RecordSystemOneStage(e.algorithm.Type, stage.Name, stage.Model, outcome, time.Since(stageStarted).Seconds())
		if qualityErr != nil {
			return Candidate{}, qualityErr
		}
		if passes {
			copy := candidate
			accepted = &copy
			if e.policy == nil {
				return candidate, nil
			}
		}
		if e.policy != nil {
			index = e.policy.Next(stage.Name, request.Features(candidate.Observations), stages, attempted, e.algorithm.Policy.CostWeight)
			if index < 0 {
				index = terminalJudge(stages, attempted, accepted, candidates)
			}
		} else {
			index++
		}
	}
	if accepted != nil {
		return *accepted, nil
	}
	return Candidate{}, ErrUnresolved
}

// A policy's judge is an operator-authored terminal fallback, not a fitted
// native action. It runs only when previous valid answers remain unresolved.
func terminalJudge(stages []config.CascadeStage, attempted map[string]bool, accepted *Candidate, candidates []Candidate) int {
	if accepted != nil || len(candidates) == 0 {
		return -1
	}
	for i, stage := range stages {
		if stage.Kind == "judge" && stage.IsEnabled() && !attempted[stage.Name] {
			return i
		}
	}
	return -1
}

func (e *Executor) accept(request *NativeRequest, stage config.CascadeStage, candidate Candidate) (bool, error) {
	if !complete(candidate.Observations) {
		return false, nil
	}
	if stage.Accept != nil && !acceptNative(candidate.Observations, stage.Accept, false) {
		return false, nil
	}
	if e.algorithm.Quality.Type == "calibrated" {
		return e.quality(request, candidate)
	}
	return acceptNative(candidate.Observations, e.algorithm.Quality.Acceptance, true), nil
}

func (e *Executor) invokeStage(ctx context.Context, request *NativeRequest, stage config.CascadeStage, candidates []Candidate, invoke Invoke) (Candidate, error) {
	if stage.Model == "" {
		return Candidate{}, errors.New("native stage has no candidate")
	}
	if stage.Timeout != "" {
		duration, err := time.ParseDuration(stage.Timeout)
		if err != nil || duration <= 0 {
			return Candidate{}, errors.New("invalid stage timeout")
		}
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, duration)
		defer cancel()
	}
	model := stage.Model
	if stage.Kind == "judge" {
		return invokeJudge(ctx, request, stage, candidates, invoke)
	}
	if stage.Kind != "native" {
		return Candidate{}, errors.New("unsupported native stage role")
	}
	input := request.Body
	if e.policy != nil || e.algorithm.Quality.Type == "calibrated" {
		input = request.InferenceBody
	}
	status, body, err := invoke(ctx, model, input)
	if err != nil {
		return Candidate{}, err
	}
	if status < http.StatusOK || status >= http.StatusMultipleChoices || !json.Valid(body) || len(body) > 4<<20 {
		return Candidate{}, fmt.Errorf("native candidate returned invalid response (status %d)", status)
	}
	return Candidate{Stage: stage.Name, Model: model, Body: body, Invocation: body}, nil
}

func complete(observations []Observation) bool {
	if len(observations) == 0 {
		return false
	}
	for _, observation := range observations {
		if !observation.Valid {
			return false
		}
	}
	return true
}
