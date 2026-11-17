package looper

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/graph"
)

// An executor's budget lowers the run's call limit to its max_calls and
// bounds the run by its deadline; the hard per-request limit still caps a
// larger max_calls.
func TestTemplateAppliesTheAlgorithmBudget(t *testing.T) {
	program, err := Template(&config.LooperConfig{}, config.DecisionAlgorithmEscalate, &config.AlgorithmBudget{Deadline: "2s", MaxCalls: 3}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if program.Limits.MaxHops != 3 || program.Limits.Timeout != 2*time.Second {
		t.Fatalf("limits = %+v, want 3 calls within 2s", program.Limits)
	}
	program, err = Template(&config.LooperConfig{}, config.DecisionAlgorithmFusion, &config.AlgorithmBudget{Deadline: "2s", MaxCalls: 100}, nil)
	if err != nil || program.Limits.MaxHops != config.MaxUpstreamCallsPerRequest {
		t.Fatalf("limits = %+v, %v, want the per-request limit", program.Limits, err)
	}
	if _, err := Template(&config.LooperConfig{}, config.DecisionAlgorithmRatings, &config.AlgorithmBudget{Deadline: "0s", MaxCalls: 1}, nil); err == nil {
		t.Fatal("a budget without a positive deadline must not build a program")
	}
}

func TestTemplateBoundsTheRunWithTheCallBudget(t *testing.T) {
	for _, algorithm := range []string{
		config.DecisionAlgorithmFusion,
		config.DecisionAlgorithmReMoM,
		config.DecisionAlgorithmWorkflows,
	} {
		t.Run(algorithm, func(t *testing.T) {
			program, err := Template(&config.LooperConfig{}, algorithm, nil, nil)
			if err != nil {
				t.Fatalf("Template: %v", err)
			}
			if program.Limits.MaxHops != config.MaxUpstreamCallsPerRequest {
				t.Fatalf("MaxHops = %d, want %d", program.Limits.MaxHops, config.MaxUpstreamCallsPerRequest)
			}
		})
	}
}

// A static workflow with one model per role spends the 32-call budget on its
// roles, then attempts final synthesis as hop 33. on_error=skip may recover
// ordinary workflow failures, but it must not turn this exhausted run into a
// successful worker fallback.
func TestTemplateWorkflowHopBudgetStaysFatalThroughFinalFallback(t *testing.T) {
	roles := make([]config.WorkflowRoleConfig, config.MaxUpstreamCallsPerRequest)
	for i := range roles {
		roles[i] = config.WorkflowRoleConfig{Name: fmt.Sprintf("role-%d", i), Models: []string{"worker"}}
	}
	program, err := Template(&config.LooperConfig{}, config.DecisionAlgorithmWorkflows, nil, nil)
	if err != nil {
		t.Fatalf("Template: %v", err)
	}
	req := &Request{
		OriginalRequest: workflowTestRequest(),
		ModelRefs:       []config.ModelRef{{Model: "worker"}},
		Algorithm: &config.AlgorithmConfig{
			Type:    config.DecisionAlgorithmWorkflows,
			OnError: config.WorkflowOnErrorSkip,
			Workflows: &config.WorkflowsAlgorithmConfig{
				Mode:        config.WorkflowModeStatic,
				Roles:       roles,
				Final:       config.WorkflowFinalConfig{Model: "worker"},
				MaxSteps:    config.MaxUpstreamCallsPerRequest,
				MaxParallel: 1,
			},
		},
	}
	input := graph.Input{Values: map[string]any{}}
	RequestValue.Put(input.Values, req)
	caller := &workflowBudgetCaller{}
	outcome, err := graph.Run(context.Background(), program, input, graph.Options{
		Caller: caller,
		Hop:    routing.Hop{Decision: "workflow-budget-test"},
	})
	if !errors.Is(err, graph.ErrHopLimit) {
		t.Fatalf("err = %v, want ErrHopLimit", err)
	}
	if outcome.Hops != config.MaxUpstreamCallsPerRequest {
		t.Fatalf("hops = %d, want %d", outcome.Hops, config.MaxUpstreamCallsPerRequest)
	}
	if caller.calls != config.MaxUpstreamCallsPerRequest {
		t.Fatalf("caller calls = %d, want %d; final fallback must not dispatch", caller.calls, config.MaxUpstreamCallsPerRequest)
	}
	if _, ok := ResponseValue.In(outcome.Values); ok {
		t.Fatal("budget exhaustion produced a Looper response")
	}
}

type workflowBudgetCaller struct{ calls int }

func (c *workflowBudgetCaller) Call(_ context.Context, req *graph.HopRequest) (*graph.HopResponse, error) {
	c.calls++
	return &graph.HopResponse{Status: 200, Body: workflowChatCompletion(req.Model, "worker")}, nil
}
