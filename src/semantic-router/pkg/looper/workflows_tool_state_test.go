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
	if err := json.Unmarshal(wireResponseForTest(t, resp), &body); err != nil {
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
	trace, ok := resp.IntermediateResponses.(*workflowTrace)
	if !ok || trace.PendingToolCall == nil {
		t.Fatalf("response metadata missing pending tool state: %#v", resp.IntermediateResponses)
	}
	pending := trace.PendingToolCall
	if pending.Model != "worker-model" {
		t.Fatalf("pending model = %v, want worker-model", pending.Model)
	}
	if pending.AgentID != "worker:0:worker-model" {
		t.Fatalf("pending agent_id = %v, want worker:0:worker-model", pending.AgentID)
	}
	if pending.StateID == "" {
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
			SemanticRequest:  workflowSemanticTestRequest(req),
			executionRequest: req,
			ModelRefs:        []config.ModelRef{{Model: "worker-model"}},
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
	assistantMessage, toolCallID := assistantToolMessageFromResponse(t, wireResponseForTest(t, firstResp))
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
	if !strings.Contains(semanticTextForTest(t, secondResp), "dynamic final answer") {
		t.Fatalf("resume response missing dynamic final answer: %s", semanticTextForTest(t, secondResp))
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
	assistantMessage, toolCallID := assistantToolMessageFromResponse(t, wireResponseForTest(t, firstResp))
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
	if !strings.Contains(semanticTextForTest(t, secondResp), "final answer after tool") {
		t.Fatalf("resume response missing final answer: %s", semanticTextForTest(t, secondResp))
	}
}

func TestWorkflowsResumeFinalReportsOnlyNewProviderUsage(t *testing.T) {
	var workerCalls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload := decodeWorkflowRequestPayload(t, r)
		w.Header().Set("Content-Type", "application/json")
		switch payload.Model {
		case "worker-model":
			workerCalls++
			if workerCalls == 1 {
				_, _ = w.Write(workflowCompletionWithUsage(t,
					workflowToolCallCompletion("worker-model", "call_lookup"),
					TokenUsage{PromptTokens: 5, CompletionTokens: 2, TotalTokens: 7},
				))
				return
			}
			_, _ = w.Write(workflowCompletionWithUsage(t,
				workflowChatCompletion("worker-model", "worker completed with tool result"),
				TokenUsage{PromptTokens: 11, CompletionTokens: 3, TotalTokens: 14},
			))
		case "verifier-model":
			_, _ = w.Write(workflowCompletionWithUsage(t,
				workflowChatCompletion("verifier-model", "final answer after tool"),
				TokenUsage{PromptTokens: 13, CompletionTokens: 4, TotalTokens: 17},
			))
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
	requireWorkflowResponseUsage(t, firstResp, TokenUsage{PromptTokens: 5, CompletionTokens: 2, TotalTokens: 7})

	assistantMessage, toolCallID := assistantToolMessageFromResponse(t, wireResponseForTest(t, firstResp))
	resumeReq := workflowToolResumeRequest(t, assistantMessage, toolCallID)
	secondResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(resumeReq))
	if err != nil {
		t.Fatalf("resume Execute failed: %v", err)
	}
	requireWorkflowResponseUsage(t, secondResp, TokenUsage{PromptTokens: 24, CompletionTokens: 7, TotalTokens: 31})
}

func TestWorkflowsRepeatedInterruptReportsOnlyEachRequestsNewProviderUsage(t *testing.T) {
	var workerCalls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload := decodeWorkflowRequestPayload(t, r)
		w.Header().Set("Content-Type", "application/json")
		switch payload.Model {
		case "worker-model":
			workerCalls++
			switch workerCalls {
			case 1:
				_, _ = w.Write(workflowCompletionWithUsage(t,
					workflowToolCallCompletion("worker-model", "call_lookup"),
					TokenUsage{PromptTokens: 5, CompletionTokens: 2, TotalTokens: 7},
				))
			case 2:
				_, _ = w.Write(workflowCompletionWithUsage(t,
					workflowToolCallCompletion("worker-model", "call_lookup"),
					TokenUsage{PromptTokens: 8, CompletionTokens: 3, TotalTokens: 11},
				))
			case 3:
				_, _ = w.Write(workflowCompletionWithUsage(t,
					workflowChatCompletion("worker-model", "worker completed"),
					TokenUsage{PromptTokens: 10, CompletionTokens: 3, TotalTokens: 13},
				))
			default:
				t.Fatalf("unexpected worker call count: %d", workerCalls)
			}
		case "verifier-model":
			_, _ = w.Write(workflowCompletionWithUsage(t,
				workflowChatCompletion("verifier-model", "final answer"),
				TokenUsage{PromptTokens: 13, CompletionTokens: 4, TotalTokens: 17},
			))
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
	requireWorkflowResponseUsage(t, firstResp, TokenUsage{PromptTokens: 5, CompletionTokens: 2, TotalTokens: 7})

	assistantMessage, firstToolCallID := assistantToolMessageFromResponse(t, wireResponseForTest(t, firstResp))
	firstResumeReq := workflowToolResumeRequest(t, assistantMessage, firstToolCallID)
	secondResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(firstResumeReq))
	if err != nil {
		t.Fatalf("second Execute failed: %v", err)
	}
	requireWorkflowResponseUsage(t, secondResp, TokenUsage{PromptTokens: 8, CompletionTokens: 3, TotalTokens: 11})

	assistantMessage, secondToolCallID := assistantToolMessageFromResponse(t, wireResponseForTest(t, secondResp))
	secondResumeReq := workflowToolResumeRequest(t, assistantMessage, secondToolCallID)
	finalResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(secondResumeReq))
	if err != nil {
		t.Fatalf("third Execute failed: %v", err)
	}
	requireWorkflowResponseUsage(t, finalResp, TokenUsage{PromptTokens: 23, CompletionTokens: 7, TotalTokens: 30})
}

func TestWorkflowsRequiresAllToolResultsBeforeResumingAgent(t *testing.T) {
	server, tracker := newWorkflowMultiToolResultServer(t)
	defer server.Close()

	looperCfg := workflowToolLooperConfig(server.URL, t.TempDir())
	firstResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(workflowToolTestRequest()))
	if err != nil {
		t.Fatalf("first Execute failed: %v", err)
	}
	assistantMessage, toolCallIDs := assistantToolMessageIDsFromResponse(t, wireResponseForTest(t, firstResp))
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
	if !strings.Contains(semanticTextForTest(t, secondResp), "final answer after two tool results") {
		t.Fatalf("resume response missing final answer: %s", semanticTextForTest(t, secondResp))
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
	assistantMessage, toolCallID := assistantToolMessageFromResponse(t, wireResponseForTest(t, firstResp))
	firstResumeReq := workflowToolResumeRequest(t, assistantMessage, toolCallID)

	secondResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(firstResumeReq))
	if err != nil {
		t.Fatalf("second Execute failed: %v", err)
	}
	secondFinish := workflowChoiceFinishReason(t, wireResponseForTest(t, secondResp))
	if secondFinish != "tool_calls" {
		t.Fatalf("second finish_reason = %v, want tool_calls", secondFinish)
	}
	assistantMessage, toolCallID = assistantToolMessageFromResponse(t, wireResponseForTest(t, secondResp))
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
	if !strings.Contains(semanticTextForTest(t, thirdResp), "final answer after two tools") {
		t.Fatalf("final response missing expected answer: %s", semanticTextForTest(t, thirdResp))
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
	assistantMessage, toolCallID := assistantToolMessageFromResponse(t, wireResponseForTest(t, firstResp))
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
	assistantMessage, firstToolCallID := assistantToolMessageFromResponse(t, wireResponseForTest(t, firstResp))
	firstResumeReq := workflowToolResumeRequest(t, assistantMessage, firstToolCallID)

	secondResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(firstResumeReq))
	if err != nil {
		t.Fatalf("second Execute failed: %v", err)
	}
	if workflowChoiceFinishReason(t, wireResponseForTest(t, secondResp)) != "tool_calls" {
		t.Fatalf("second response did not request another tool: %s", wireResponseForTest(t, secondResp))
	}
	assistantMessage, secondToolCallID := assistantToolMessageFromResponse(t, wireResponseForTest(t, secondResp))
	if secondToolCallID == firstToolCallID {
		t.Fatalf("tool_call_id was reused across tool turns: %q", secondToolCallID)
	}

	staleResumeReq := workflowToolResumeRequest(t, assistantMessage, firstToolCallID)
	_, err = NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolStateValidationRequest(staleResumeReq))
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
	assistantMessage, firstToolCallID := assistantToolMessageFromResponse(t, wireResponseForTest(t, firstResp))
	firstState := workflowStoredPendingState(t, stateDir, firstToolCallID)
	assertWorkflowInitialPendingToolState(t, firstState)

	firstResumeReq := workflowToolResumeRequest(t, assistantMessage, firstToolCallID)
	secondResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(firstResumeReq))
	if err != nil {
		t.Fatalf("second Execute failed: %v", err)
	}
	assistantMessage, secondToolCallID := assistantToolMessageFromResponse(t, wireResponseForTest(t, secondResp))
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
	trace, ok := firstResp.IntermediateResponses.(*workflowTrace)
	if !ok || trace.PendingToolCall == nil {
		t.Fatalf("response metadata missing pending tool state: %#v", firstResp.IntermediateResponses)
	}
	pending := trace.PendingToolCall
	if pending.Phase != workflowToolPhaseFinal {
		t.Fatalf("pending phase = %v, want final", pending.Phase)
	}
	if pending.Model != "verifier-model" {
		t.Fatalf("pending model = %v, want verifier-model", pending.Model)
	}
	assistantMessage, toolCallID := assistantToolMessageFromResponse(t, wireResponseForTest(t, firstResp))
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
	if !strings.Contains(semanticTextForTest(t, secondResp), "final answer after final tool") {
		t.Fatalf("resume response missing final answer: %s", semanticTextForTest(t, secondResp))
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
	assistantMessage, toolCallID := assistantToolMessageFromResponse(t, wireResponseForTest(t, firstResp))
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
