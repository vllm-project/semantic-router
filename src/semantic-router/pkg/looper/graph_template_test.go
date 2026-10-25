package looper

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/graph"
)

func TestTemplateBoundsTheRunWithTheCallBudget(t *testing.T) {
	for _, algorithm := range []string{
		config.DecisionAlgorithmFusion,
		config.DecisionAlgorithmReMoM,
		config.DecisionAlgorithmWorkflows,
	} {
		t.Run(algorithm, func(t *testing.T) {
			program, err := Template(&config.LooperConfig{}, algorithm, nil)
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
	program, err := Template(&config.LooperConfig{}, config.DecisionAlgorithmWorkflows, nil)
	if err != nil {
		t.Fatalf("Template: %v", err)
	}
	req := &Request{
		OriginalRequest: workflowTestRequest(),
		ModelRefs:       []config.ModelRef{{Model: "worker"}},
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmWorkflows,
			Workflows: &config.WorkflowsAlgorithmConfig{
				Mode:        config.WorkflowModeStatic,
				Roles:       roles,
				Final:       config.WorkflowFinalConfig{Model: "worker"},
				MaxSteps:    config.MaxUpstreamCallsPerRequest,
				MaxParallel: 1,
				OnError:     config.WorkflowOnErrorSkip,
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
