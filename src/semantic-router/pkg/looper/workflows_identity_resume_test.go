package looper

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestWorkflowsResumeRejectsPersistedAmbiguousIdentities(t *testing.T) {
	for _, id := range []string{"lookup", "lookup:0:worker-model"} {
		for _, policy := range []string{config.WorkflowOnErrorFail, config.WorkflowOnErrorSkip} {
			t.Run(id+"/"+policy, func(t *testing.T) {
				testWorkflowResumeAmbiguousIdentity(t, id, policy)
			})
		}
	}
}

func testWorkflowResumeAmbiguousIdentity(t *testing.T, conflictingID, policy string) {
	t.Helper()
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		payload := decodeWorkflowRequestPayload(t, r)
		w.Header().Set("Content-Type", "application/json")
		switch payload.Model {
		case "planner":
			_, _ = w.Write(workflowChatCompletion("planner", `{"steps":[{"id":"lookup","models":["worker-model"],"prompt":"Use lookup."}]}`))
		case "worker-model":
			if payloadHasToolMessage(payload.Messages) {
				_, _ = w.Write(workflowChatCompletion("worker-model", "completed"))
			} else {
				_, _ = w.Write(workflowToolCallCompletion("worker-model", "call_lookup"))
			}
		case "verifier-model":
			_, _ = w.Write(workflowChatCompletion("verifier-model", "final answer"))
		default:
			http.Error(w, "unexpected model", http.StatusBadRequest)
		}
	}))
	defer server.Close()

	cfg := workflowToolLooperConfig(server.URL, t.TempDir())
	looper := NewWorkflowsLooper(cfg)
	req := workflowToolLooperRequest(workflowToolTestRequest())
	req.Algorithm.Workflows.Mode = config.WorkflowModeDynamic
	req.Algorithm.Workflows.Roles = nil
	req.Algorithm.Workflows.Planner = config.WorkflowPlannerConfig{Model: "planner"}
	req.Algorithm.Workflows.OnError = policy
	ctx := context.Background()
	first, err := looper.Execute(ctx, req)
	if err != nil {
		t.Fatalf("initial execution: %v", err)
	}
	assistant, callID := assistantToolMessageFromResponse(t, first.Body)
	stateID, ok := parseWorkflowToolStateID(callID)
	if !ok {
		t.Fatal("missing continuation ID")
	}
	recipe := config.DefaultRecipeName
	claim, ok, err := looper.toolStates.Claim(ctx, recipe, stateID)
	if err != nil || !ok || claim == nil {
		t.Fatalf("load paused state: found=%v, err=%v", ok, err)
	}
	state := claim.State
	// Emulate a plan persisted by a version that accepted ambiguous IDs.
	state.Plan.Steps = append(state.Plan.Steps, workflowPlanStep{
		ID: conflictingID, Models: []string{"worker-model"}, Prompt: "Another task.",
	})
	if replaceErr := looper.toolStates.Replace(ctx, recipe, stateID, claim.Token, state); replaceErr != nil {
		t.Fatalf("persist legacy plan: %v", replaceErr)
	}
	before := calls.Load()
	req.OriginalRequest = workflowToolResumeRequest(t, assistant, callID)
	// A new instance must load and revalidate the serialized plan.
	_, err = NewWorkflowsLooper(cfg).Execute(ctx, req)
	if err == nil || !strings.Contains(err.Error(), "access identity") {
		t.Fatalf("expected identity rejection on resume, got %v", err)
	}
	if calls.Load() != before {
		t.Fatalf("resume dispatched upstream calls: before=%d, after=%d", before, calls.Load())
	}
	// A validation failure must not consume the continuation state.
	restored, found, claimErr := looper.toolStates.Claim(ctx, recipe, stateID)
	if claimErr != nil || !found || restored == nil {
		t.Fatalf("rejected state was not restored: found=%v, err=%v", found, claimErr)
	}
	if releaseErr := looper.toolStates.Release(ctx, recipe, stateID, restored.Token); releaseErr != nil {
		t.Fatalf("release restored state claim: %v", releaseErr)
	}
}
