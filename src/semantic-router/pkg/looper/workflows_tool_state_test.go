package looper

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestWorkflowsWorkerToolCallReturnsPendingFlowState(t *testing.T) {
	var calls []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		calls = append(calls, payload.Model)
		if payload.Model != "worker-model" {
			t.Fatalf("unexpected model call before tool resume: %s", payload.Model)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(workflowToolCallCompletion("worker-model", "call_lookup"))
	}))
	defer server.Close()

	resp, err := NewWorkflowsLooper(workflowToolLooperConfig(server.URL, t.TempDir())).Execute(context.Background(), workflowToolLooperRequest(workflowToolTestRequest()))
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	if len(calls) != 1 || calls[0] != "worker-model" {
		t.Fatalf("calls = %v, want only worker-model", calls)
	}

	var body map[string]interface{}
	if err := json.Unmarshal(resp.Body, &body); err != nil {
		t.Fatalf("response body is not JSON: %v", err)
	}
	choice := body["choices"].([]interface{})[0].(map[string]interface{})
	if choice["finish_reason"] != "tool_calls" {
		t.Fatalf("finish_reason = %v, want tool_calls", choice["finish_reason"])
	}
	message := choice["message"].(map[string]interface{})
	toolCall := message["tool_calls"].([]interface{})[0].(map[string]interface{})
	toolCallID := toolCall["id"].(string)
	if !strings.HasPrefix(toolCallID, workflowToolCallIDPrefix) {
		t.Fatalf("tool_call id %q missing workflow prefix", toolCallID)
	}
	flow := body["flow"].(map[string]interface{})
	pending := flow["pending_tool_call"].(map[string]interface{})
	if pending["model"] != "worker-model" {
		t.Fatalf("pending model = %v, want worker-model", pending["model"])
	}
	if pending["agent_id"] != "worker:0:worker-model" {
		t.Fatalf("pending agent_id = %v, want worker:0:worker-model", pending["agent_id"])
	}
	if pending["state_id"] == "" {
		t.Fatalf("pending state id missing: %#v", pending)
	}
}

func TestWorkflowsDynamicResumesWorkerToolCallAndSynthesizes(t *testing.T) {
	var workerSawToolResult bool
	var finalSawWorkerAnswer bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload := decodeWorkflowRequestPayload(t, r)
		w.Header().Set("Content-Type", "application/json")
		switch payload.Model {
		case "qwen-coordinator":
			if payloadMessagesContain(payload.Messages, "worker completed with dynamic tool") {
				finalSawWorkerAnswer = true
				_, _ = w.Write(workflowChatCompletion("qwen-coordinator", "dynamic final answer"))
				return
			}
			_, _ = w.Write(workflowChatCompletion("qwen-coordinator", `{"steps":[{"id":"lookup","role":"worker","models":["worker-model"],"prompt":"Use lookup when needed, then solve."}],"final":{"prompt":"merge worker evidence"}}`))
		case "worker-model":
			if payloadHasToolMessage(payload.Messages) {
				workerSawToolResult = true
				_, _ = w.Write(workflowChatCompletion("worker-model", "worker completed with dynamic tool"))
				return
			}
			_, _ = w.Write(workflowToolCallCompletion("worker-model", "call_lookup"))
		default:
			t.Fatalf("unexpected model call: %s", payload.Model)
		}
	}))
	defer server.Close()

	looperCfg := workflowToolLooperConfig(server.URL, t.TempDir())
	makeRequest := func(req *openai.ChatCompletionNewParams) *Request {
		includeTrace := true
		return &Request{
			OriginalRequest: req,
			ModelRefs:       []config.ModelRef{{Model: "worker-model"}},
			Algorithm: &config.AlgorithmConfig{
				Type: "workflows",
				Workflows: &config.WorkflowsAlgorithmConfig{
					Mode:                         config.WorkflowModeDynamic,
					Planner:                      config.WorkflowPlannerConfig{Model: "qwen-coordinator"},
					MaxSteps:                     2,
					MaxParallel:                  1,
					IncludeIntermediateResponses: &includeTrace,
				},
			},
			DecisionName: "dynamic-tool-flow-test",
		}
	}

	firstResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), makeRequest(workflowToolTestRequest()))
	if err != nil {
		t.Fatalf("first Execute failed: %v", err)
	}
	assistantMessage, toolCallID := assistantToolMessageFromResponse(t, firstResp.Body)
	resumeReq := workflowToolResumeRequest(t, assistantMessage, toolCallID)

	secondResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), makeRequest(resumeReq))
	if err != nil {
		t.Fatalf("resume Execute failed: %v", err)
	}
	if !workerSawToolResult {
		t.Fatal("dynamic worker resume request did not include tool result")
	}
	if !finalSawWorkerAnswer {
		t.Fatal("dynamic final synthesis did not receive resumed worker output")
	}
	if !strings.Contains(string(secondResp.Body), "dynamic final answer") {
		t.Fatalf("resume response missing dynamic final answer: %s", string(secondResp.Body))
	}
}

func TestWorkflowsLegacyFunctionCallReturnsPendingFlowState(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if payload["model"] != "worker-model" {
			t.Fatalf("unexpected model call before legacy function resume: %s", payload["model"])
		}
		if !payloadHasTopLevelField(payload, "functions") || !payloadHasTopLevelField(payload, "function_call") {
			t.Fatalf("worker missing legacy function schema: %#v", payload)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(workflowLegacyFunctionCallCompletion("worker-model"))
	}))
	defer server.Close()

	resp, err := NewWorkflowsLooper(workflowToolLooperConfig(server.URL, t.TempDir())).Execute(context.Background(), workflowToolLooperRequest(workflowLegacyFunctionTestRequest()))
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	var body map[string]interface{}
	if err := json.Unmarshal(resp.Body, &body); err != nil {
		t.Fatalf("response body is not JSON: %v", err)
	}
	choice := body["choices"].([]interface{})[0].(map[string]interface{})
	if choice["finish_reason"] != "tool_calls" {
		t.Fatalf("finish_reason = %v, want tool_calls", choice["finish_reason"])
	}
	message := choice["message"].(map[string]interface{})
	if _, ok := message["function_call"]; ok {
		t.Fatalf("legacy function_call leaked after normalization: %s", string(resp.Body))
	}
	toolCall := message["tool_calls"].([]interface{})[0].(map[string]interface{})
	toolCallID := toolCall["id"].(string)
	if !strings.HasPrefix(toolCallID, workflowToolCallIDPrefix) {
		t.Fatalf("tool_call id %q missing workflow prefix", toolCallID)
	}
	flow := body["flow"].(map[string]interface{})
	pending := flow["pending_tool_call"].(map[string]interface{})
	if pending["model"] != "worker-model" {
		t.Fatalf("pending model = %v, want worker-model", pending["model"])
	}
}

func TestWorkflowsResumesWorkerToolCallAndContinuesToFinal(t *testing.T) {
	var workerSawToolResult bool
	var finalCalled bool
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
		case "worker-model":
			if payloadHasToolMessage(payload.Messages) {
				workerSawToolResult = true
				_, _ = w.Write(workflowChatCompletion("worker-model", "worker completed with tool result"))
				return
			}
			_, _ = w.Write(workflowToolCallCompletion("worker-model", "call_lookup"))
		case "verifier-model":
			finalCalled = true
			_, _ = w.Write(workflowChatCompletion("verifier-model", "final answer after tool"))
		default:
			t.Fatalf("unexpected model call: %s", payload.Model)
		}
	}))
	defer server.Close()

	looperCfg := workflowToolLooperConfig(server.URL, t.TempDir())
	looperReq := workflowToolLooperRequest(workflowToolTestRequest())
	firstResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), looperReq)
	if err != nil {
		t.Fatalf("first Execute failed: %v", err)
	}
	assistantMessage, toolCallID := assistantToolMessageFromResponse(t, firstResp.Body)
	resumeReq := workflowToolResumeRequest(t, assistantMessage, toolCallID)

	secondResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(resumeReq))
	if err != nil {
		t.Fatalf("resume Execute failed: %v", err)
	}
	if !workerSawToolResult {
		t.Fatal("worker resume request did not include tool result")
	}
	if !finalCalled {
		t.Fatal("final synthesis was not called after worker tool result")
	}
	if !strings.Contains(string(secondResp.Body), "final answer after tool") {
		t.Fatalf("resume response missing final answer: %s", string(secondResp.Body))
	}
}

func TestWorkflowsRequiresAllToolResultsBeforeResumingAgent(t *testing.T) {
	server, tracker := newWorkflowMultiToolResultServer(t)
	defer server.Close()

	looperCfg := workflowToolLooperConfig(server.URL, t.TempDir())
	firstResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(workflowToolTestRequest()))
	if err != nil {
		t.Fatalf("first Execute failed: %v", err)
	}
	assistantMessage, toolCallIDs := assistantToolMessageIDsFromResponse(t, firstResp.Body)
	if len(toolCallIDs) != 2 {
		t.Fatalf("tool_call_ids = %v, want 2 ids", toolCallIDs)
	}

	partialResumeReq := workflowToolResumeRequestWithIDs(t, assistantMessage, toolCallIDs[0])
	_, err = NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(partialResumeReq))
	if err == nil || !strings.Contains(err.Error(), "missing tool result") {
		t.Fatalf("expected missing tool result error, got %v", err)
	}
	if tracker.workerCalls != 1 {
		t.Fatalf("partial resume should not call worker; calls = %d", tracker.workerCalls)
	}

	fullResumeReq := workflowToolResumeRequestWithIDs(t, assistantMessage, toolCallIDs...)
	secondResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(fullResumeReq))
	if err != nil {
		t.Fatalf("full resume Execute failed: %v", err)
	}
	if !tracker.workerSawBothTools {
		t.Fatal("worker resume request did not include both pending tool results")
	}
	if !tracker.finalCalled {
		t.Fatal("final synthesis was not called after full tool result set")
	}
	if !strings.Contains(string(secondResp.Body), "final answer after two tool results") {
		t.Fatalf("resume response missing final answer: %s", string(secondResp.Body))
	}
}

func TestWorkflowsWorkerSupportsMultipleToolTurnsBeforeContinuing(t *testing.T) {
	server, tracker := newWorkflowMultiTurnToolServer(t)
	defer server.Close()

	looperCfg := workflowToolLooperConfig(server.URL, t.TempDir())
	firstResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(workflowToolTestRequest()))
	if err != nil {
		t.Fatalf("first Execute failed: %v", err)
	}
	assistantMessage, toolCallID := assistantToolMessageFromResponse(t, firstResp.Body)
	firstResumeReq := workflowToolResumeRequest(t, assistantMessage, toolCallID)

	secondResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(firstResumeReq))
	if err != nil {
		t.Fatalf("second Execute failed: %v", err)
	}
	secondFinish := workflowChoiceFinishReason(t, secondResp.Body)
	if secondFinish != "tool_calls" {
		t.Fatalf("second finish_reason = %v, want tool_calls", secondFinish)
	}
	assistantMessage, toolCallID = assistantToolMessageFromResponse(t, secondResp.Body)
	secondResumeReq := workflowToolResumeRequest(t, assistantMessage, toolCallID)

	thirdResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(secondResumeReq))
	if err != nil {
		t.Fatalf("third Execute failed: %v", err)
	}
	if tracker.workerCalls != 3 {
		t.Fatalf("worker calls = %d, want 3", tracker.workerCalls)
	}
	if !tracker.finalCalled {
		t.Fatal("final synthesis was not called after multi-turn tool loop")
	}
	if !strings.Contains(string(thirdResp.Body), "final answer after two tools") {
		t.Fatalf("final response missing expected answer: %s", string(thirdResp.Body))
	}
}

func TestWorkflowsToolStateIsConsumeOnceAfterSuccessfulResume(t *testing.T) {
	var workerCalls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload := decodeWorkflowRequestPayload(t, r)
		w.Header().Set("Content-Type", "application/json")
		switch payload.Model {
		case "worker-model":
			workerCalls++
			if payloadHasToolMessage(payload.Messages) {
				_, _ = w.Write(workflowChatCompletion("worker-model", "worker completed once"))
				return
			}
			_, _ = w.Write(workflowToolCallCompletion("worker-model", "call_lookup"))
		case "verifier-model":
			_, _ = w.Write(workflowChatCompletion("verifier-model", "final answer"))
		default:
			t.Fatalf("unexpected model call: %s", payload.Model)
		}
	}))
	defer server.Close()

	looperCfg := workflowToolLooperConfig(server.URL, t.TempDir())
	firstResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(workflowToolTestRequest()))
	if err != nil {
		t.Fatalf("first Execute failed: %v", err)
	}
	assistantMessage, toolCallID := assistantToolMessageFromResponse(t, firstResp.Body)
	resumeReq := workflowToolResumeRequest(t, assistantMessage, toolCallID)

	_, err = NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(resumeReq))
	if err != nil {
		t.Fatalf("resume Execute failed: %v", err)
	}
	_, err = NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(resumeReq))
	if err == nil || !strings.Contains(err.Error(), "not found or expired") {
		t.Fatalf("expected consumed state error, got %v", err)
	}
	if workerCalls != 2 {
		t.Fatalf("reused consumed state should not call worker again; calls = %d", workerCalls)
	}
}

func TestWorkflowsWorkerRepeatedBackendToolCallIDsAreDistinctAcrossTurns(t *testing.T) {
	server, _ := newWorkflowRepeatedToolIDServer(t, "worker completed after repeated tool id")
	defer server.Close()

	looperCfg := workflowToolLooperConfig(server.URL, t.TempDir())
	firstResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(workflowToolTestRequest()))
	if err != nil {
		t.Fatalf("first Execute failed: %v", err)
	}
	assistantMessage, firstToolCallID := assistantToolMessageFromResponse(t, firstResp.Body)
	firstResumeReq := workflowToolResumeRequest(t, assistantMessage, firstToolCallID)

	secondResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(firstResumeReq))
	if err != nil {
		t.Fatalf("second Execute failed: %v", err)
	}
	if workflowChoiceFinishReason(t, secondResp.Body) != "tool_calls" {
		t.Fatalf("second response did not request another tool: %s", string(secondResp.Body))
	}
	assistantMessage, secondToolCallID := assistantToolMessageFromResponse(t, secondResp.Body)
	if secondToolCallID == firstToolCallID {
		t.Fatalf("tool_call_id was reused across tool turns: %q", secondToolCallID)
	}

	staleResumeReq := workflowToolResumeRequest(t, assistantMessage, firstToolCallID)
	_, err = NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(staleResumeReq))
	if err == nil || !strings.Contains(err.Error(), "was not requested") {
		t.Fatalf("expected stale tool result rejection, got %v", err)
	}

	secondResumeReq := workflowToolResumeRequest(t, assistantMessage, secondToolCallID)
	_, err = NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(secondResumeReq))
	if err != nil {
		t.Fatalf("third Execute failed: %v", err)
	}
}

func TestWorkflowsPersistsAgentToolTrajectoryAcrossToolTurns(t *testing.T) {
	server, _ := newWorkflowRepeatedToolIDServer(t, "worker completed")
	defer server.Close()

	stateDir := t.TempDir()
	looperCfg := workflowToolLooperConfig(server.URL, stateDir)
	firstResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(workflowToolTestRequest()))
	if err != nil {
		t.Fatalf("first Execute failed: %v", err)
	}
	assistantMessage, firstToolCallID := assistantToolMessageFromResponse(t, firstResp.Body)
	firstState := workflowStoredPendingState(t, stateDir, firstToolCallID)
	assertWorkflowInitialPendingToolState(t, firstState)

	firstResumeReq := workflowToolResumeRequest(t, assistantMessage, firstToolCallID)
	secondResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(firstResumeReq))
	if err != nil {
		t.Fatalf("second Execute failed: %v", err)
	}
	assistantMessage, secondToolCallID := assistantToolMessageFromResponse(t, secondResp.Body)
	secondState := workflowStoredPendingState(t, stateDir, secondToolCallID)
	assertWorkflowSecondPendingToolState(t, secondState, firstToolCallID)

	secondResumeReq := workflowToolResumeRequest(t, assistantMessage, secondToolCallID)
	_, err = NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(secondResumeReq))
	if err != nil {
		t.Fatalf("third Execute failed: %v", err)
	}
}

func TestWorkflowsResumesFinalToolCallWithoutRerunningWorkers(t *testing.T) {
	server, tracker := newWorkflowFinalToolServer(t)
	defer server.Close()

	looperCfg := workflowToolLooperConfig(server.URL, t.TempDir())
	firstResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(workflowToolTestRequest()))
	if err != nil {
		t.Fatalf("first Execute failed: %v", err)
	}
	if tracker.workerCalls != 1 {
		t.Fatalf("worker calls before resume = %d, want 1", tracker.workerCalls)
	}
	var firstBody map[string]interface{}
	if unmarshalErr := json.Unmarshal(firstResp.Body, &firstBody); unmarshalErr != nil {
		t.Fatalf("response body is not JSON: %v", unmarshalErr)
	}
	flow := firstBody["flow"].(map[string]interface{})
	pending := flow["pending_tool_call"].(map[string]interface{})
	if pending["phase"] != workflowToolPhaseFinal {
		t.Fatalf("pending phase = %v, want final", pending["phase"])
	}
	if pending["model"] != "verifier-model" {
		t.Fatalf("pending model = %v, want verifier-model", pending["model"])
	}
	assistantMessage, toolCallID := assistantToolMessageFromResponse(t, firstResp.Body)
	resumeReq := workflowToolResumeRequest(t, assistantMessage, toolCallID)

	secondResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(resumeReq))
	if err != nil {
		t.Fatalf("resume Execute failed: %v", err)
	}
	if tracker.workerCalls != 1 {
		t.Fatalf("worker was rerun during final tool resume; calls = %d", tracker.workerCalls)
	}
	if tracker.finalCalls != 2 {
		t.Fatalf("final calls = %d, want 2", tracker.finalCalls)
	}
	if !tracker.finalSawWorkerAnswer {
		t.Fatal("final agent did not receive workflow outputs before tool call")
	}
	if !tracker.finalSawToolResult {
		t.Fatal("final resume request did not include tool result")
	}
	if !strings.Contains(string(secondResp.Body), "final answer after final tool") {
		t.Fatalf("resume response missing final answer: %s", string(secondResp.Body))
	}
}

func TestWorkflowsAccessListExposesPriorOutputWithoutToolTrajectory(t *testing.T) {
	var consumerSawLookupOutput bool
	var consumerSawLookupToolTrajectory bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload := decodeWorkflowRequestPayload(t, r)
		w.Header().Set("Content-Type", "application/json")
		switch payload.Model {
		case "worker-a":
			if payloadHasToolMessage(payload.Messages) {
				_, _ = w.Write(workflowChatCompletion("worker-a", "lookup output visible through access list"))
				return
			}
			_, _ = w.Write(workflowToolCallCompletion("worker-a", "call_lookup_a"))
		case "worker-b":
			consumerSawLookupOutput = payloadMessagesContain(payload.Messages, "lookup output visible through access list")
			consumerSawLookupToolTrajectory = payloadHasToolMessage(payload.Messages) || payloadHasAssistantToolCalls(payload.Messages)
			_, _ = w.Write(workflowChatCompletion("worker-b", "consumer used allowed output"))
		case "verifier-model":
			_, _ = w.Write(workflowChatCompletion("verifier-model", "final answer"))
		default:
			t.Fatalf("unexpected model call: %s", payload.Model)
		}
	}))
	defer server.Close()

	looperCfg := workflowToolLooperConfig(server.URL, t.TempDir())
	firstResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowAccessListToolLooperRequest(workflowToolTestRequest()))
	if err != nil {
		t.Fatalf("first Execute failed: %v", err)
	}
	assistantMessage, toolCallID := assistantToolMessageFromResponse(t, firstResp.Body)
	resumeReq := workflowToolResumeRequest(t, assistantMessage, toolCallID)

	_, err = NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowAccessListToolLooperRequest(resumeReq))
	if err != nil {
		t.Fatalf("resume Execute failed: %v", err)
	}
	if !consumerSawLookupOutput {
		t.Fatal("access-list consumer did not receive prior agent output")
	}
	if consumerSawLookupToolTrajectory {
		t.Fatal("access-list consumer received prior agent private tool trajectory")
	}
}

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
	assistantMessage, toolCallID := assistantToolMessageFromResponse(t, firstResp.Body)
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
	assistantMessage, toolCallID := assistantToolMessageFromResponse(t, firstResp.Body)
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
	if !strings.Contains(string(secondResp.Body), "final answer") {
		t.Fatalf("resume response missing final answer: %s", string(secondResp.Body))
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
	assistantMessage, toolCallID := assistantToolMessageFromResponse(t, firstResp.Body)
	firstResumeReq := workflowToolResumeRequest(t, assistantMessage, toolCallID)

	secondResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowParallelToolLooperRequest(firstResumeReq))
	if err != nil {
		t.Fatalf("second Execute failed: %v", err)
	}
	if workflowChoiceFinishReason(t, secondResp.Body) != "tool_calls" {
		t.Fatalf("second worker did not return a tool call: %s", string(secondResp.Body))
	}
	assistantMessage, toolCallID = assistantToolMessageFromResponse(t, secondResp.Body)
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
	assistantMessage, firstToolCallID := assistantToolMessageFromResponse(t, firstResp.Body)
	firstResumeReq := workflowToolResumeRequest(t, assistantMessage, firstToolCallID)

	secondResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowParallelToolLooperRequest(firstResumeReq))
	if err != nil {
		t.Fatalf("second Execute failed: %v", err)
	}
	if workflowChoiceFinishReason(t, secondResp.Body) != "tool_calls" {
		t.Fatalf("second worker did not return a tool call: %s", string(secondResp.Body))
	}
	assistantMessage, secondToolCallID := assistantToolMessageFromResponse(t, secondResp.Body)
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
	assertWorkflowTraceToolTrajectory(t, finalResp.Body, "workers:0:worker-a", firstToolCallID)
	assertWorkflowTraceToolTrajectory(t, finalResp.Body, "workers:1:worker-b", secondToolCallID)
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
	assistantMessage, toolCallID := assistantToolMessageFromResponse(t, firstResp.Body)
	otherToolCallID := strings.Replace(toolCallID, "call_lookup", "call_other", 1)
	resumeReq := workflowToolResumeRequest(t, assistantMessage, otherToolCallID)

	_, err = NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(resumeReq))
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
	assistantMessage, toolCallID := assistantToolMessageFromResponse(t, firstResp.Body)
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

	_, err = NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(resumeReq))
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
	assistantMessage, toolCallID := assistantToolMessageFromResponse(t, firstResp.Body)
	resumeReq := workflowToolResumeRequest(t, assistantMessage, toolCallID)

	_, err = NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowTwoStepToolLooperRequest(resumeReq))
	if err != nil {
		t.Fatalf("resume Execute failed: %v", err)
	}
	if !finalSawThinkerEvidence {
		t.Fatal("final synthesis did not receive prior step output after file-state resume")
	}
}
