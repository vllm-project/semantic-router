package looper

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestWorkflowsDynamicAmbiguousIdentityPolicy(t *testing.T) {
	for _, secondID := range []string{"research", "research:0:worker"} {
		for _, policy := range []string{config.WorkflowOnErrorFail, config.WorkflowOnErrorSkip} {
			t.Run(secondID+"/"+policy, func(t *testing.T) {
				testWorkflowAmbiguousIdentityPolicy(t, secondID, policy)
			})
		}
	}
}

func testWorkflowAmbiguousIdentityPolicy(t *testing.T, secondID, policy string) {
	t.Helper()
	plan := &workflowPlan{Steps: []workflowPlanStep{
		{ID: "research", Models: []string{"worker"}, Prompt: "INVALID_PLAN_STEP"},
		{ID: secondID, Models: []string{"worker"}, Prompt: "INVALID_PLAN_STEP"},
	}}
	planJSON, err := json.Marshal(plan)
	if err != nil {
		t.Fatal(err)
	}
	var plannerCalls, workerCalls, finalCalls, invalidCalls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload := decodeWorkflowRequestPayload(t, r)
		messages, _ := json.Marshal(payload.Messages)
		if strings.Contains(string(messages), "INVALID_PLAN_STEP") {
			invalidCalls.Add(1)
		}
		w.Header().Set("Content-Type", "application/json")
		switch payload.Model {
		case "planner":
			plannerCalls.Add(1)
			_, _ = w.Write(workflowChatCompletion("planner", string(planJSON)))
		case "worker":
			workerCalls.Add(1)
			_, _ = w.Write(workflowChatCompletion("worker", "worker answer"))
		case "final":
			finalCalls.Add(1)
			_, _ = w.Write(workflowChatCompletion("final", "final answer"))
		default:
			http.Error(w, "unexpected model", http.StatusBadRequest)
		}
	}))
	defer server.Close()
	resp, err := NewWorkflowsLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), &Request{
		OriginalRequest: workflowTestRequest(),
		ModelRefs:       []config.ModelRef{{Model: "worker"}, {Model: "final"}},
		Algorithm: &config.AlgorithmConfig{Type: "workflows", Workflows: &config.WorkflowsAlgorithmConfig{
			Mode:     config.WorkflowModeDynamic,
			Planner:  config.WorkflowPlannerConfig{Model: "planner"},
			Final:    config.WorkflowFinalConfig{Model: "final"},
			MaxSteps: 3, MaxParallel: 1, OnError: policy,
		}},
		DecisionName: "identity-test",
	})
	if plannerCalls.Load() != 1 || invalidCalls.Load() != 0 {
		t.Fatalf("planner calls=%d, invalid plan calls=%d", plannerCalls.Load(), invalidCalls.Load())
	}
	if policy == config.WorkflowOnErrorFail {
		if err == nil || !strings.Contains(err.Error(), "access identity") {
			t.Fatalf("expected collision error, got %v", err)
		}
		if workerCalls.Load() != 0 || finalCalls.Load() != 0 {
			t.Fatal("invalid plan dispatched worker or final calls")
		}
		return
	}
	if err != nil {
		t.Fatalf("fallback failed: %v", err)
	}
	if workerCalls.Load() != 1 || finalCalls.Load() != 1 {
		t.Fatalf("fallback worker calls=%d, final calls=%d", workerCalls.Load(), finalCalls.Load())
	}
	trace, ok := resp.IntermediateResponses.(*workflowTrace)
	if !ok || trace.Plan == nil || len(trace.Plan.Steps) != 1 || trace.Plan.Steps[0].ID != "fallback_solve" {
		t.Fatalf("expected fallback plan, got %#v", resp.IntermediateResponses)
	}
}
