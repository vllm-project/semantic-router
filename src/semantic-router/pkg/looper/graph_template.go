package looper

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/graph"
)

// StepType is the request-graph step that runs a Looper algorithm. A
// decision's built-in template runs it, and custom graphs embed it.
const StepType = "looper"

// RequestValue and ResponseValue carry a Looper execution through its run:
// the request the step executes, and the response it produced, with the
// accounting, replay and trace evidence the caller records.
var (
	RequestValue  = graph.NewKey[*Request]("looper.request")
	ResponseValue = graph.NewKey[*Response]("looper.response")
)

// ErrNoRequest reports a Looper step in a run that carries no Looper request.
var ErrNoRequest = errors.New("looper: the run carries no Looper request")

// algorithmStep runs one Looper algorithm. Its model calls are hops of the
// run, so the run's budget, cancellation, spans and evidence cover them, and
// its answer is the step's one result.
type algorithmStep struct {
	cfg       *config.LooperConfig
	algorithm string
	workflows *WorkflowStateService
}

// Run executes the algorithm on the run's Looper request.
func (s *algorithmStep) Run(ctx context.Context, x *graph.Exec, st *graph.State) error {
	req, ok := RequestValue.Get(st)
	if !ok || req == nil {
		return ErrNoRequest
	}
	looper, err := FactoryWithClientAndWorkflowState(s.cfg, s.algorithm, NewHopClient(s.cfg, x), s.workflows)
	if err != nil {
		return err
	}
	resp, err := ExecuteWithLatency(ctx, looper, req)
	if err != nil {
		return err
	}
	ResponseValue.Set(st, resp)
	st.Results = []*graph.Result{{
		Step:   s.algorithm,
		Model:  resp.Model,
		Status: http.StatusOK,
		Header: routing.Header{{Name: "content-type", Value: resp.ContentType}},
		Body:   resp.Body,
	}}
	return nil
}

// Template returns the built-in request graph of a Looper algorithm: the
// algorithm's step, then a respond step that answers with its result. The
// configuration and the workflow state belong to the router generation that
// runs the template; limit is the decision's algorithm budget, if it has one.
func Template(cfg *config.LooperConfig, algorithm string, limit *config.AlgorithmBudget, workflows *WorkflowStateService) (*graph.Program, error) {
	if _, err := constructorFor(algorithm); err != nil {
		return nil, err
	}
	limits, err := templateLimits(limit)
	if err != nil {
		return nil, err
	}
	return &graph.Program{
		Name: "looper." + algorithm,
		Steps: graph.Sequence{
			{ID: algorithm, Type: StepType, Node: &algorithmStep{cfg: cfg, algorithm: algorithm, workflows: workflows}},
			{ID: "respond", Type: graph.TypeRespond, Node: &graph.Respond{}},
		},
		Limits: limits,
	}, nil
}

// templateLimits bounds a run. The hard per-request call budget rejects
// configurations that are already over it at config load, so a statically
// valid program only reaches it through call counts that are knowable during
// execution (retries, nested and multi-stage fan-out). An algorithm budget
// lowers the call limit to its max_calls and adds its deadline. A run that
// would cross a limit fails closed instead of amplifying past it.
func templateLimits(limit *config.AlgorithmBudget) (graph.Limits, error) {
	limits := graph.Limits{MaxHops: config.MaxUpstreamCallsPerRequest}
	if limit == nil {
		return limits, nil
	}
	deadline, err := time.ParseDuration(limit.Deadline)
	if err != nil || deadline <= 0 || limit.MaxCalls <= 0 {
		return graph.Limits{}, fmt.Errorf("invalid algorithm budget: deadline %q, max_calls %d", limit.Deadline, limit.MaxCalls)
	}
	limits.Timeout = deadline
	limits.MaxHops = min(limits.MaxHops, limit.MaxCalls)
	return limits, nil
}
