package systemone

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/budget"
)

// ErrUnresolved means no complete response satisfied the declared quality
// conditions within the available stages and request budget.
var ErrUnresolved = errors.New("native inference quality unresolved")

// BudgetExhaustedError reports a native request stopped by its shared deadline
// or physical inference-call limit.
type BudgetExhaustedError struct {
	Calls  int
	Reason string
	Cause  error
}

func (e *BudgetExhaustedError) Error() string { return "native request budget exhausted" }

func (e *BudgetExhaustedError) Unwrap() error { return e.Cause }

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
	quality   QualityEvaluator
}

func NewExecutor(algorithm *config.AlgorithmConfig, quality QualityEvaluator) (*Executor, error) {
	if !algorithm.IsNative() {
		return nil, errors.New("invalid native execution plan")
	}
	e := &Executor{algorithm: algorithm, quality: quality}
	if algorithm.NativeQuality().EffectiveType() == config.EscalateQualityCalibrated && quality == nil {
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
		var incapable *IncapableError
		if resultErr != nil {
			outcome = "unresolved"
			if errors.As(resultErr, &incapable) {
				outcome = "incapable"
			} else if errors.Is(resultErr, budget.ErrExhausted) || errors.Is(resultErr, context.DeadlineExceeded) {
				outcome = "budget_exhausted"
			} else if ctx.Err() != nil {
				outcome = "canceled"
			}
		}
		metrics.RecordSystemOneRequest(e.algorithm.Type, outcome, time.Since(started).Seconds())
	}()
	skipped, refusal := e.incapableStages(ctx, request, membersFrom(ctx))
	if refusal != nil {
		return Candidate{}, refusal
	}
	ledger, hasLedger := budget.Current(ctx)
	var candidates []Candidate
	var history []StageEvidence
	for index, stage := range e.algorithm.NativeStages() {
		if err := ctx.Err(); err != nil {
			return Candidate{}, err
		}
		if !stage.IsEnabled() {
			continue
		}
		if skipped[index] {
			metrics.RecordSystemOneStageSkipped(e.algorithm.Type, stage.StageName(), stage.Model, "incapable")
			continue
		}
		stageStarted := time.Now()
		callsBefore := 0
		if hasLedger {
			callsBefore = ledger.Used()
		}
		candidate, err := e.invokeStage(ctx, request, stage, candidates, invoke)
		name := stage.StageName()
		recordStageCalls := func() {
			if hasLedger {
				metrics.RecordSystemOneStageCalls(e.algorithm.Type, name, stage.Model, ledger.Used()-callsBefore)
			}
		}
		evidence := StageEvidence{Stage: name, Model: stage.Model, Response: candidate.Invocation, Outcome: "response"}
		if err != nil {
			evidence.Outcome = "error"
		}
		history = append(history, evidence)
		candidate.History = append([]StageEvidence(nil), history...)
		if err != nil {
			if ctx.Err() != nil {
				metrics.RecordSystemOneStage(e.algorithm.Type, name, stage.Model, "canceled", time.Since(stageStarted).Seconds())
				recordStageCalls()
				return Candidate{}, ctx.Err()
			}
			if errors.Is(err, budget.ErrExhausted) {
				metrics.RecordSystemOneStage(e.algorithm.Type, name, stage.Model, "budget_exhausted", time.Since(stageStarted).Seconds())
				recordStageCalls()
				return Candidate{}, err
			}
			metrics.RecordSystemOneStage(e.algorithm.Type, name, stage.Model, "error", time.Since(stageStarted).Seconds())
			recordStageCalls()
			continue
		}
		candidate.Observations = request.Observe(candidate.Body)
		if complete(candidate.Observations) {
			candidates = append(candidates, candidate)
		}
		passes, qualityErr := e.accept(request, stage, candidate)
		if contextErr := ctx.Err(); contextErr != nil {
			metrics.RecordSystemOneStage(e.algorithm.Type, name, stage.Model, "canceled", time.Since(stageStarted).Seconds())
			recordStageCalls()
			return Candidate{}, contextErr
		}
		outcome := "rejected"
		if qualityErr != nil {
			outcome = "evaluation_error"
		} else if passes {
			outcome = "accepted"
		} else if !complete(candidate.Observations) {
			outcome = "invalid"
		}
		metrics.RecordSystemOneStage(e.algorithm.Type, name, stage.Model, outcome, time.Since(stageStarted).Seconds())
		recordStageCalls()
		if qualityErr != nil {
			return Candidate{}, qualityErr
		}
		if passes {
			return candidate, nil
		}
	}
	return Candidate{}, ErrUnresolved
}

// incapableStages are the enabled native stages, by index, whose model
// provably cannot answer the request: its card, listing or declared
// capabilities lack a question type, a modality or full input the request
// needs. A model nothing describes is asked as before, and so is every model
// for a request every System One model answers. When every native stage is
// incapable, the request goes to none of them.
func (e *Executor) incapableStages(ctx context.Context, request *NativeRequest, members MemberCapabilities) (map[int]bool, *IncapableError) {
	if members == nil || !request.Requirements.beyondStandard() {
		return nil, nil
	}
	stages := e.algorithm.NativeStages()
	type found struct {
		capabilities Capabilities
		known, asked bool
	}
	verdicts := make([]found, len(stages))
	var wg sync.WaitGroup
	for index, stage := range stages {
		if !stage.IsEnabled() || stage.StageKind() != config.EscalateStageNative {
			continue
		}
		verdicts[index].asked = true
		wg.Add(1)
		go func() {
			defer wg.Done()
			verdicts[index].capabilities, verdicts[index].known = members(ctx, stage.Model)
		}()
	}
	wg.Wait()
	skipped := map[int]bool{}
	refusal := &IncapableError{}
	native := 0
	for index, stage := range stages {
		if !verdicts[index].asked {
			continue
		}
		native++
		capabilities, known := verdicts[index].capabilities, verdicts[index].known
		if !known {
			continue
		}
		missing := capabilities.Missing(request.Requirements)
		if len(missing) == 0 {
			continue
		}
		skipped[index] = true
		for _, need := range missing {
			if !slices.Contains(refusal.Missing, need) {
				refusal.Missing = append(refusal.Missing, need)
			}
		}
		if !slices.ContainsFunc(refusal.Members, func(member IncapableMember) bool { return member.Model == stage.Model }) {
			refusal.Members = append(refusal.Members, IncapableMember{Model: stage.Model, Source: capabilities.Source, Missing: missing})
		}
	}
	if native > 0 && len(skipped) == native {
		return nil, refusal
	}
	return skipped, nil
}

// accept applies the stage's own rules, then the quality policy: the
// calibration evidence, or the whole-bundle rules of an uncalibrated policy,
// which a complete answer passes when there are none. Without either, the
// stage's rules judge the whole bundle; beside one, they only add conditions.
func (e *Executor) accept(request *NativeRequest, stage config.EscalateStage, candidate Candidate) (bool, error) {
	if !complete(candidate.Observations) {
		return false, nil
	}
	quality := e.algorithm.NativeQuality()
	calibrated := quality.EffectiveType() == config.EscalateQualityCalibrated
	whole := !calibrated && (quality == nil || quality.Accept == nil)
	if stage.Accept != nil && !acceptNative(candidate.Observations, stage.Accept, whole) {
		return false, nil
	}
	if calibrated {
		return e.quality(request, candidate)
	}
	if quality == nil || quality.Accept == nil {
		return true, nil
	}
	return acceptNative(candidate.Observations, quality.Accept, true), nil
}

func (e *Executor) invokeStage(ctx context.Context, request *NativeRequest, stage config.EscalateStage, candidates []Candidate, invoke Invoke) (candidate Candidate, invokeErr error) {
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
	// Buffered responses can finish decoding after cancellation. Check before
	// the stage context is released so neither native nor judge results outlive
	// their deadline; an expired stage may still advance within the parent budget.
	defer func() {
		if contextErr := ctx.Err(); contextErr != nil {
			candidate, invokeErr = Candidate{}, contextErr
		}
	}()
	model := stage.Model
	switch stage.StageKind() {
	case config.EscalateStageJudge:
		return invokeJudge(ctx, request, stage, candidates, invoke)
	case config.EscalateStageNative:
	default:
		return Candidate{}, errors.New("unsupported native stage role")
	}
	input := request.Body
	if e.algorithm.NativeQuality().EffectiveType() == config.EscalateQualityCalibrated {
		input = request.InferenceBody
	}
	status, body, err := invoke(ctx, model, input)
	if err != nil {
		return Candidate{}, err
	}
	if status < http.StatusOK || status >= http.StatusMultipleChoices || !json.Valid(body) || len(body) > 4<<20 {
		return Candidate{}, fmt.Errorf("native candidate returned invalid response (status %d)", status)
	}
	return Candidate{Stage: stage.StageName(), Model: model, Body: body, Invocation: body}, nil
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
