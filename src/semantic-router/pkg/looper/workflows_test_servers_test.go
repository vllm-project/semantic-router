package looper

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

type workflowToolPrivacyTracker struct {
	workerBSawToolTrajectory bool
	verifierCalled           bool
}

type workflowRequestPayload struct {
	Model               string                   `json:"model"`
	MaxCompletionTokens *int64                   `json:"max_completion_tokens,omitempty"`
	Messages            []map[string]interface{} `json:"messages"`
}

func decodeWorkflowRequestPayload(t *testing.T, r *http.Request) workflowRequestPayload {
	t.Helper()
	var payload workflowRequestPayload
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	return payload
}

func newWorkflowToolPrivacyServer(t *testing.T) (*httptest.Server, *workflowToolPrivacyTracker) {
	t.Helper()
	tracker := &workflowToolPrivacyTracker{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload := decodeWorkflowRequestPayload(t, r)
		w.Header().Set("Content-Type", "application/json")
		switch payload.Model {
		case "worker-a":
			writeWorkflowWorkerAResponse(w, payload.Messages)
		case "worker-b":
			tracker.workerBSawToolTrajectory = payloadHasToolMessage(payload.Messages) || payloadHasAssistantToolCalls(payload.Messages)
			_, _ = w.Write(workflowChatCompletion("worker-b", "worker-b clean context"))
		case "verifier-model":
			tracker.verifierCalled = true
			_, _ = w.Write(workflowChatCompletion("verifier-model", "final answer"))
		default:
			t.Fatalf("unexpected model call: %s", payload.Model)
		}
	}))
	return server, tracker
}

func writeWorkflowWorkerAResponse(w http.ResponseWriter, messages []map[string]interface{}) {
	if payloadHasToolMessage(messages) {
		_, _ = w.Write(workflowChatCompletion("worker-a", "worker-a completed privately"))
		return
	}
	_, _ = w.Write(workflowToolCallCompletion("worker-a", "call_lookup"))
}

func newWorkflowToolSchemaServer(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		writeWorkflowToolSchemaResponse(t, w, r, payload)
	}))
}

func writeWorkflowToolSchemaResponse(
	t *testing.T,
	w http.ResponseWriter,
	r *http.Request,
	payload map[string]interface{},
) {
	t.Helper()
	w.Header().Set("Content-Type", "application/json")
	model, _ := payload["model"].(string)
	switch model {
	case "qwen-coordinator":
		writeWorkflowPlannerToolSchemaResponse(t, w, r, payload)
	case "worker-a":
		assertWorkflowPayloadHasCallableSchemas(t, payload)
		_, _ = w.Write(workflowChatCompletion("worker-a", "worker answer"))
	default:
		t.Fatalf("unexpected model call: %s", model)
	}
}

func writeWorkflowPlannerToolSchemaResponse(
	t *testing.T,
	w http.ResponseWriter,
	r *http.Request,
	payload map[string]interface{},
) {
	t.Helper()
	if r.Header.Get("x-vsr-looper-iteration") != "1" {
		_, _ = w.Write(workflowChatCompletion("qwen-coordinator", "final answer"))
		return
	}
	assertWorkflowPayloadHasNoCallableSchemas(t, payload)
	_, _ = w.Write(workflowChatCompletion("qwen-coordinator", `{"steps":[{"id":"solve","role":"worker","models":["worker-a"],"prompt":"solve with tools if needed"}],"final":{"prompt":"merge"}}`))
}

func assertWorkflowPayloadHasNoCallableSchemas(t *testing.T, payload map[string]interface{}) {
	t.Helper()
	for _, field := range []string{"tools", "tool_choice", "functions", "function_call"} {
		if payloadHasTopLevelField(payload, field) {
			t.Fatalf("planner received %s: %#v", field, payload[field])
		}
	}
}

func assertWorkflowPayloadHasCallableSchemas(t *testing.T, payload map[string]interface{}) {
	t.Helper()
	for _, field := range []string{"tools", "tool_choice", "functions", "function_call"} {
		if !payloadHasTopLevelField(payload, field) {
			t.Fatalf("worker missing %s in tool-capable step: %#v", field, payload)
		}
	}
}

type workflowMultiToolResultTracker struct {
	workerCalls        int
	workerSawBothTools bool
	finalCalled        bool
}

func newWorkflowMultiToolResultServer(t *testing.T) (*httptest.Server, *workflowMultiToolResultTracker) {
	t.Helper()
	tracker := &workflowMultiToolResultTracker{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload := decodeWorkflowRequestPayload(t, r)
		w.Header().Set("Content-Type", "application/json")
		tracker.writeResponse(t, w, payload)
	}))
	return server, tracker
}

func (tracker *workflowMultiToolResultTracker) writeResponse(
	t *testing.T,
	w http.ResponseWriter,
	payload workflowRequestPayload,
) {
	t.Helper()
	switch payload.Model {
	case "worker-model":
		tracker.writeWorkerResponse(w, payload.Messages)
	case "verifier-model":
		tracker.finalCalled = true
		_, _ = w.Write(workflowChatCompletion("verifier-model", "final answer after two tool results"))
	default:
		t.Fatalf("unexpected model call: %s", payload.Model)
	}
}

func (tracker *workflowMultiToolResultTracker) writeWorkerResponse(
	w http.ResponseWriter,
	messages []map[string]interface{},
) {
	tracker.workerCalls++
	if payloadHasToolMessage(messages) {
		tracker.workerSawBothTools = payloadToolMessageIDContains(messages, "call_alpha") &&
			payloadToolMessageIDContains(messages, "call_beta")
		_, _ = w.Write(workflowChatCompletion("worker-model", "worker completed with two tool results"))
		return
	}
	_, _ = w.Write(workflowMultipleToolCallCompletion("worker-model", "call_alpha", "call_beta"))
}

type workflowMultiTurnToolTracker struct {
	workerCalls int
	finalCalled bool
}

func newWorkflowMultiTurnToolServer(t *testing.T) (*httptest.Server, *workflowMultiTurnToolTracker) {
	t.Helper()
	tracker := &workflowMultiTurnToolTracker{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload := decodeWorkflowRequestPayload(t, r)
		w.Header().Set("Content-Type", "application/json")
		tracker.writeResponse(t, w, payload)
	}))
	return server, tracker
}

func (tracker *workflowMultiTurnToolTracker) writeResponse(
	t *testing.T,
	w http.ResponseWriter,
	payload workflowRequestPayload,
) {
	t.Helper()
	switch payload.Model {
	case "worker-model":
		tracker.writeWorkerResponse(t, w, payload.Messages)
	case "verifier-model":
		tracker.finalCalled = true
		_, _ = w.Write(workflowChatCompletion("verifier-model", "final answer after two tools"))
	default:
		t.Fatalf("unexpected model call: %s", payload.Model)
	}
}

func (tracker *workflowMultiTurnToolTracker) writeWorkerResponse(
	t *testing.T,
	w http.ResponseWriter,
	messages []map[string]interface{},
) {
	t.Helper()
	tracker.workerCalls++
	switch tracker.workerCalls {
	case 1:
		assertWorkflowToolMessageAbsent(t, messages, "first worker call")
		_, _ = w.Write(workflowToolCallCompletion("worker-model", "call_first"))
	case 2:
		assertWorkflowToolMessageContains(t, messages, "call_first", "second worker call")
		_, _ = w.Write(workflowToolCallCompletion("worker-model", "call_second"))
	case 3:
		assertWorkflowToolMessageContains(t, messages, "call_second", "third worker call")
		_, _ = w.Write(workflowChatCompletion("worker-model", "worker completed after two tools"))
	default:
		t.Fatalf("unexpected worker call count: %d", tracker.workerCalls)
	}
}

func assertWorkflowToolMessageAbsent(t *testing.T, messages []map[string]interface{}, context string) {
	t.Helper()
	if payloadHasToolMessage(messages) {
		t.Fatalf("%s unexpectedly had a tool result", context)
	}
}

func assertWorkflowToolMessageContains(
	t *testing.T,
	messages []map[string]interface{},
	needle string,
	context string,
) {
	t.Helper()
	if !payloadToolMessageIDContains(messages, needle) {
		t.Fatalf("%s missing %s tool result: %#v", context, needle, messages)
	}
}

type workflowRepeatedToolIDTracker struct {
	workerCalls int
	workerDone  string
}

func newWorkflowRepeatedToolIDServer(t *testing.T, workerDone string) (*httptest.Server, *workflowRepeatedToolIDTracker) {
	t.Helper()
	tracker := &workflowRepeatedToolIDTracker{workerDone: workerDone}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload := decodeWorkflowRequestPayload(t, r)
		w.Header().Set("Content-Type", "application/json")
		tracker.writeResponse(t, w, payload)
	}))
	return server, tracker
}

func (tracker *workflowRepeatedToolIDTracker) writeResponse(
	t *testing.T,
	w http.ResponseWriter,
	payload workflowRequestPayload,
) {
	t.Helper()
	switch payload.Model {
	case "worker-model":
		tracker.writeWorkerResponse(t, w)
	case "verifier-model":
		_, _ = w.Write(workflowChatCompletion("verifier-model", "final answer"))
	default:
		t.Fatalf("unexpected model call: %s", payload.Model)
	}
}

func (tracker *workflowRepeatedToolIDTracker) writeWorkerResponse(t *testing.T, w http.ResponseWriter) {
	t.Helper()
	tracker.workerCalls++
	switch tracker.workerCalls {
	case 1, 2:
		_, _ = w.Write(workflowToolCallCompletion("worker-model", "call_lookup"))
	case 3:
		_, _ = w.Write(workflowChatCompletion("worker-model", tracker.workerDone))
	default:
		t.Fatalf("unexpected worker call count: %d", tracker.workerCalls)
	}
}

type workflowFinalToolTracker struct {
	workerCalls          int
	finalCalls           int
	finalSawToolResult   bool
	finalSawWorkerAnswer bool
}

func newWorkflowFinalToolServer(t *testing.T) (*httptest.Server, *workflowFinalToolTracker) {
	t.Helper()
	tracker := &workflowFinalToolTracker{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload := decodeWorkflowRequestPayload(t, r)
		w.Header().Set("Content-Type", "application/json")
		tracker.writeResponse(t, w, payload)
	}))
	return server, tracker
}

func (tracker *workflowFinalToolTracker) writeResponse(
	t *testing.T,
	w http.ResponseWriter,
	payload workflowRequestPayload,
) {
	t.Helper()
	switch payload.Model {
	case "worker-model":
		tracker.workerCalls++
		_, _ = w.Write(workflowChatCompletion("worker-model", "worker evidence for final"))
	case "verifier-model":
		tracker.writeFinalResponse(w, payload.Messages)
	default:
		t.Fatalf("unexpected model call: %s", payload.Model)
	}
}

func (tracker *workflowFinalToolTracker) writeFinalResponse(
	w http.ResponseWriter,
	messages []map[string]interface{},
) {
	tracker.finalCalls++
	tracker.finalSawWorkerAnswer = tracker.finalSawWorkerAnswer ||
		payloadMessagesContain(messages, "worker evidence for final")
	if payloadHasToolMessage(messages) {
		tracker.finalSawToolResult = true
		_, _ = w.Write(workflowChatCompletion("verifier-model", "final answer after final tool"))
		return
	}
	_, _ = w.Write(workflowToolCallCompletion("verifier-model", "call_lookup"))
}

type workflowSecondWorkerToolTracker struct {
	workerBFirstSawPriorTool  bool
	workerBResumeSawOwnTool   bool
	workerBResumeSawOtherTool bool
}

func newWorkflowSecondWorkerToolServer(t *testing.T) (*httptest.Server, *workflowSecondWorkerToolTracker) {
	t.Helper()
	tracker := &workflowSecondWorkerToolTracker{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload := decodeWorkflowRequestPayload(t, r)
		w.Header().Set("Content-Type", "application/json")
		tracker.writeResponse(t, w, payload)
	}))
	return server, tracker
}

func (tracker *workflowSecondWorkerToolTracker) writeResponse(
	t *testing.T,
	w http.ResponseWriter,
	payload workflowRequestPayload,
) {
	t.Helper()
	switch payload.Model {
	case "worker-a":
		writeWorkflowWorkerAResponse(w, payload.Messages)
	case "worker-b":
		tracker.writeWorkerBResponse(w, payload.Messages)
	case "verifier-model":
		_, _ = w.Write(workflowChatCompletion("verifier-model", "final answer"))
	default:
		t.Fatalf("unexpected model call: %s", payload.Model)
	}
}

func (tracker *workflowSecondWorkerToolTracker) writeWorkerBResponse(
	w http.ResponseWriter,
	messages []map[string]interface{},
) {
	if payloadHasToolMessage(messages) {
		tracker.workerBResumeSawOwnTool = payloadToolMessageIDContains(messages, "call_lookup_b")
		tracker.workerBResumeSawOtherTool = payloadToolMessageIDContains(messages, "call_lookup_a")
		_, _ = w.Write(workflowChatCompletion("worker-b", "worker-b completed privately"))
		return
	}
	tracker.workerBFirstSawPriorTool = payloadHasAssistantToolCalls(messages) ||
		payloadToolMessageIDContains(messages, "call_lookup_a")
	_, _ = w.Write(workflowToolCallCompletion("worker-b", "call_lookup_b"))
}
