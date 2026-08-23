package looper

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestWorkflowsRejectsToolResumeForDifferentDecision(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(workflowToolCallCompletion("worker-model", "call_lookup"))
	}))
	defer server.Close()

	looperCfg := workflowToolLooperConfig(server.URL, t.TempDir())
	firstResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(workflowToolTestRequest()))
	if err != nil {
		t.Fatalf("first Execute failed: %v", err)
	}
	assistantMessage, toolCallID := assistantToolMessageFromResponse(t, wireResponseForTest(t, firstResp))
	resumeReq := workflowToolResumeRequest(t, assistantMessage, toolCallID)
	resumeLooperReq := workflowToolLooperRequest(resumeReq)
	resumeLooperReq.DecisionName = "different-decision"

	_, err = NewWorkflowsLooper(looperCfg).Execute(context.Background(), resumeLooperReq)
	if err == nil || !strings.Contains(err.Error(), "belongs to decision") {
		t.Fatalf("expected decision mismatch error, got %v", err)
	}
}

func TestWorkflowsToolTrajectoryStaysPrivateToWorker(t *testing.T) {
	server, tracker := newWorkflowToolPrivacyServer(t)
	defer server.Close()

	looperCfg := workflowToolLooperConfig(server.URL, t.TempDir())
	firstReq := workflowParallelToolLooperRequest(workflowToolTestRequest())
	firstResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), firstReq)
	if err != nil {
		t.Fatalf("first Execute failed: %v", err)
	}
	assistantMessage, toolCallID := assistantToolMessageFromResponse(t, wireResponseForTest(t, firstResp))
	resumeReq := workflowToolResumeRequest(t, assistantMessage, toolCallID)

	secondResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowParallelToolLooperRequest(resumeReq))
	if err != nil {
		t.Fatalf("resume Execute failed: %v", err)
	}
	if tracker.workerBSawToolTrajectory {
		t.Fatal("worker-b saw worker-a private tool trajectory")
	}
	if !tracker.verifierCalled {
		t.Fatal("final verifier was not called")
	}
	if !strings.Contains(semanticTextForTest(t, secondResp), "final answer") {
		t.Fatalf("resume response missing final answer: %s", semanticTextForTest(t, secondResp))
	}
}

func TestWorkflowsSecondWorkerToolLoopUsesOwnAgentState(t *testing.T) {
	server, tracker := newWorkflowSecondWorkerToolServer(t)
	defer server.Close()

	looperCfg := workflowToolLooperConfig(server.URL, t.TempDir())
	firstResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowParallelToolLooperRequest(workflowToolTestRequest()))
	if err != nil {
		t.Fatalf("first Execute failed: %v", err)
	}
	assistantMessage, toolCallID := assistantToolMessageFromResponse(t, wireResponseForTest(t, firstResp))
	firstResumeReq := workflowToolResumeRequest(t, assistantMessage, toolCallID)

	secondResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowParallelToolLooperRequest(firstResumeReq))
	if err != nil {
		t.Fatalf("second Execute failed: %v", err)
	}
	if workflowChoiceFinishReason(t, wireResponseForTest(t, secondResp)) != "tool_calls" {
		t.Fatalf("second worker did not return a tool call: %s", wireResponseForTest(t, secondResp))
	}
	assistantMessage, toolCallID = assistantToolMessageFromResponse(t, wireResponseForTest(t, secondResp))
	secondResumeReq := workflowToolResumeRequest(t, assistantMessage, toolCallID)

	_, err = NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowParallelToolLooperRequest(secondResumeReq))
	if err != nil {
		t.Fatalf("third Execute failed: %v", err)
	}
	if tracker.workerBFirstSawPriorTool {
		t.Fatal("worker-b first call saw worker-a private tool trajectory")
	}
	if !tracker.workerBResumeSawOwnTool {
		t.Fatal("worker-b resume did not receive its own tool result")
	}
	if tracker.workerBResumeSawOtherTool {
		t.Fatal("worker-b resume received worker-a tool result")
	}
}

func TestWorkflowsPersistsPriorAgentToolTrajectoryWhenLaterAgentInterrupts(t *testing.T) {
	server, _ := newWorkflowSecondWorkerToolServer(t)
	defer server.Close()

	stateDir := t.TempDir()
	looperCfg := workflowToolLooperConfig(server.URL, stateDir)
	firstResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowParallelToolLooperRequest(workflowToolTestRequest()))
	if err != nil {
		t.Fatalf("first Execute failed: %v", err)
	}
	assistantMessage, firstToolCallID := assistantToolMessageFromResponse(t, wireResponseForTest(t, firstResp))
	firstResumeReq := workflowToolResumeRequest(t, assistantMessage, firstToolCallID)

	secondResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowParallelToolLooperRequest(firstResumeReq))
	if err != nil {
		t.Fatalf("second Execute failed: %v", err)
	}
	if workflowChoiceFinishReason(t, wireResponseForTest(t, secondResp)) != "tool_calls" {
		t.Fatalf("second worker did not return a tool call: %s", wireResponseForTest(t, secondResp))
	}
	assistantMessage, secondToolCallID := assistantToolMessageFromResponse(t, wireResponseForTest(t, secondResp))
	secondState := workflowStoredPendingState(t, stateDir, secondToolCallID)
	if secondState.AgentID != "workers:1:worker-b" {
		t.Fatalf("second state agent_id = %q, want workers:1:worker-b", secondState.AgentID)
	}
	if len(secondState.ToolTrajectory) != 0 {
		t.Fatalf("worker-b state already has own trajectory = %d turns, want 0", len(secondState.ToolTrajectory))
	}
	workerATurns := secondState.CurrentStepToolTrajectories["workers:0:worker-a"]
	if len(workerATurns) != 1 {
		t.Fatalf("worker-a current step trajectory = %d turns, want 1", len(workerATurns))
	}
	assertWorkflowToolTrajectoryTurnForAgent(t, workerATurns[0], firstToolCallID, "workers:0:worker-a")
	if _, ok := secondState.CurrentStepToolTrajectories["workers:1:worker-b"]; ok {
		t.Fatal("worker-b current step trajectory should not be populated before its tool result returns")
	}

	secondResumeReq := workflowToolResumeRequest(t, assistantMessage, secondToolCallID)
	finalResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowParallelToolLooperRequest(secondResumeReq))
	if err != nil {
		t.Fatalf("third Execute failed: %v", err)
	}
	assertWorkflowTraceToolTrajectory(t, finalResp, "workers:0:worker-a", firstToolCallID)
	assertWorkflowTraceToolTrajectory(t, finalResp, "workers:1:worker-b", secondToolCallID)
}

func TestWorkflowsRejectsToolResultForDifferentPendingCall(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(workflowToolCallCompletion("worker-model", "call_lookup"))
	}))
	defer server.Close()

	looperCfg := workflowToolLooperConfig(server.URL, t.TempDir())
	firstResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(workflowToolTestRequest()))
	if err != nil {
		t.Fatalf("first Execute failed: %v", err)
	}
	assistantMessage, toolCallID := assistantToolMessageFromResponse(t, wireResponseForTest(t, firstResp))
	otherToolCallID := strings.Replace(toolCallID, "call_lookup", "call_other", 1)
	resumeReq := workflowToolResumeRequest(t, assistantMessage, otherToolCallID)

	_, err = NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolStateValidationRequest(resumeReq))
	if err == nil || !strings.Contains(err.Error(), "was not requested") {
		t.Fatalf("expected pending tool_call_id validation error, got %v", err)
	}
}

func TestWorkflowsRejectsMixedToolResultsFromDifferentState(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(workflowToolCallCompletion("worker-model", "call_lookup"))
	}))
	defer server.Close()

	looperCfg := workflowToolLooperConfig(server.URL, t.TempDir())
	firstResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(workflowToolTestRequest()))
	if err != nil {
		t.Fatalf("first Execute failed: %v", err)
	}
	assistantMessage, toolCallID := assistantToolMessageFromResponse(t, wireResponseForTest(t, firstResp))
	resumeReq := workflowToolResumeRequest(t, assistantMessage, toolCallID)
	reqMap, ok := requestAsMap(resumeReq)
	if !ok {
		t.Fatal("resume request did not marshal to map")
	}
	messages, ok := reqMap["messages"].([]interface{})
	if !ok {
		t.Fatalf("resume request messages have unexpected type: %T", reqMap["messages"])
	}
	messages = append(messages, map[string]interface{}{
		"role":         "tool",
		"tool_call_id": workflowToolCallIDPrefix + "different_state" + workflowToolCallIDSeparator + "call_other",
		"content":      `{"value":"wrong-agent"}`,
	})
	reqMap["messages"] = messages
	data, err := json.Marshal(reqMap)
	if err != nil {
		t.Fatalf("marshal mixed resume request: %v", err)
	}
	if unmarshalErr := json.Unmarshal(data, resumeReq); unmarshalErr != nil {
		t.Fatalf("parse mixed resume request: %v", unmarshalErr)
	}

	_, err = NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolStateValidationRequest(resumeReq))
	if err == nil || !strings.Contains(err.Error(), "belongs to workflow state") {
		t.Fatalf("expected mixed workflow state validation error, got %v", err)
	}
}

func TestWorkflowsPersistentStateKeepsPriorStepOutputs(t *testing.T) {
	var finalSawThinkerEvidence bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model    string                   `json:"model"`
			Messages []map[string]interface{} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		switch payload.Model {
		case "thinker-model":
			_, _ = w.Write(workflowChatCompletion("thinker-model", "thinker evidence"))
		case "worker-model":
			if payloadHasToolMessage(payload.Messages) {
				_, _ = w.Write(workflowChatCompletion("worker-model", "worker completed with tool"))
				return
			}
			_, _ = w.Write(workflowToolCallCompletion("worker-model", "call_lookup"))
		case "verifier-model":
			finalSawThinkerEvidence = payloadMessagesContain(payload.Messages, "thinker evidence")
			_, _ = w.Write(workflowChatCompletion("verifier-model", "final answer"))
		default:
			t.Fatalf("unexpected model call: %s", payload.Model)
		}
	}))
	defer server.Close()

	looperCfg := workflowToolLooperConfig(server.URL, t.TempDir())
	firstResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowTwoStepToolLooperRequest(workflowToolTestRequest()))
	if err != nil {
		t.Fatalf("first Execute failed: %v", err)
	}
	assistantMessage, toolCallID := assistantToolMessageFromResponse(t, wireResponseForTest(t, firstResp))
	resumeReq := workflowToolResumeRequest(t, assistantMessage, toolCallID)

	_, err = NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowTwoStepToolLooperRequest(resumeReq))
	if err != nil {
		t.Fatalf("resume Execute failed: %v", err)
	}
	if !finalSawThinkerEvidence {
		t.Fatal("final synthesis did not receive prior step output after file-state resume")
	}
}
