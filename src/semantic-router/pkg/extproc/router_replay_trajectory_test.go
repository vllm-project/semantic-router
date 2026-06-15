package extproc

import (
	"testing"
	"time"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// newTrajectoryTestRouter builds a router with two records sharing a session ID.
// The first record has a user_input + assistant_tool_call + client_tool_result.
// The second record has an assistant_final_response.
func newTrajectoryTestRouter(t *testing.T) (*OpenAIRouter, string) {
	t.Helper()
	sessionID := "sess-abc123"
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))

	_, err := recorder.AddRecord(routerreplay.RoutingRecord{
		ID:        "traj-1",
		RequestID: sessionID,
		Timestamp: time.Unix(1, 0).UTC(),
		ToolTrace: &routerreplay.ToolTrace{
			Steps: []routerreplay.ToolTraceStep{
				{Type: replayToolStepUserInput, Role: "user", Text: "what is the weather?"},
				{Type: replayToolStepAssistantToolCall, Role: "assistant", ToolName: "get_weather", ToolCallID: "call-1", Arguments: `{"city":"NYC"}`},
				{Type: replayToolStepClientToolResult, Role: "tool", Text: "sunny", ToolCallID: "call-1"},
			},
		},
	})
	if err != nil {
		t.Fatalf("failed to add record: %v", err)
	}

	_, err = recorder.AddRecord(routerreplay.RoutingRecord{
		ID:        "traj-2",
		RequestID: sessionID,
		Timestamp: time.Unix(2, 0).UTC(),
		ToolTrace: &routerreplay.ToolTrace{
			Steps: []routerreplay.ToolTraceStep{
				{Type: replayToolStepAssistantFinalResponse, Role: "assistant", Text: "It is sunny in NYC."},
			},
		},
	})
	if err != nil {
		t.Fatalf("failed to add record: %v", err)
	}

	return &OpenAIRouter{
		ReplayRecorders: map[string]*routerreplay.Recorder{"default": recorder},
	}, sessionID
}

func TestHandleRouterReplayTrajectoryConvertsToolTraceToOpenAIMessages(t *testing.T) {
	router, sessionID := newTrajectoryTestRouter(t)

	response := router.handleRouterReplayAPI("GET", "/v1/router_replay/trajectory?session_id="+sessionID)
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatal("expected immediate trajectory response")
	}

	body := decodeJSONBody(t, response.GetImmediateResponse().Body)
	assertStringField(t, body, "object", "router_replay.trajectory")
	assertStringField(t, body, "session_id", sessionID)
	assertIntField(t, body, "record_count", 2)

	messages := mustTrajectoryMessages(t, body)
	if len(messages) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(messages))
	}

	assertStringField(t, messages[0], "role", "user")
	assertStringField(t, messages[0], "content", "what is the weather?")

	assertStringField(t, messages[1], "role", "assistant")
	toolCalls := mustTrajectoryToolCalls(t, messages[1])
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(toolCalls))
	}
	assertStringField(t, toolCalls[0], "id", "call-1")
	assertStringField(t, toolCalls[0], "type", "function")
	fn := mustTrajectoryFunction(t, toolCalls[0])
	assertStringField(t, fn, "name", "get_weather")
	assertStringField(t, fn, "arguments", `{"city":"NYC"}`)

	assertStringField(t, messages[2], "role", "tool")
	assertStringField(t, messages[2], "content", "sunny")
	assertStringField(t, messages[2], "tool_call_id", "call-1")

	assertStringField(t, messages[3], "role", "assistant")
	assertStringField(t, messages[3], "content", "It is sunny in NYC.")
}

func TestHandleRouterReplayTrajectoryCoalescesConsecutiveToolCalls(t *testing.T) {
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	sessionID := "sess-multi-tool"

	_, err := recorder.AddRecord(routerreplay.RoutingRecord{
		ID:        "traj-multi",
		RequestID: sessionID,
		Timestamp: time.Unix(1, 0).UTC(),
		ToolTrace: &routerreplay.ToolTrace{
			Steps: []routerreplay.ToolTraceStep{
				{Type: replayToolStepUserInput, Role: "user", Text: "run both tools"},
				{Type: replayToolStepAssistantToolCall, Role: "assistant", ToolName: "tool_a", ToolCallID: "call-a", Arguments: `{}`},
				{Type: replayToolStepAssistantToolCall, Role: "assistant", ToolName: "tool_b", ToolCallID: "call-b", Arguments: `{}`},
				{Type: replayToolStepClientToolResult, Role: "tool", Text: "result-a", ToolCallID: "call-a"},
				{Type: replayToolStepClientToolResult, Role: "tool", Text: "result-b", ToolCallID: "call-b"},
				{Type: replayToolStepAssistantFinalResponse, Role: "assistant", Text: "done"},
			},
		},
	})
	if err != nil {
		t.Fatalf("failed to add record: %v", err)
	}

	router := &OpenAIRouter{
		ReplayRecorders: map[string]*routerreplay.Recorder{"default": recorder},
	}

	response := router.handleRouterReplayAPI("GET", "/v1/router_replay/trajectory?session_id="+sessionID)
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatal("expected immediate trajectory response")
	}

	body := decodeJSONBody(t, response.GetImmediateResponse().Body)
	messages := mustTrajectoryMessages(t, body)

	// Expect: user, assistant(2 tool_calls), tool, tool, assistant(final) = 5 messages
	if len(messages) != 5 {
		t.Fatalf("expected 5 messages after coalescing, got %d", len(messages))
	}

	assertStringField(t, messages[0], "role", "user")

	assertStringField(t, messages[1], "role", "assistant")
	toolCalls := mustTrajectoryToolCalls(t, messages[1])
	if len(toolCalls) != 2 {
		t.Fatalf("expected 2 coalesced tool calls, got %d", len(toolCalls))
	}
	assertStringField(t, toolCalls[0], "id", "call-a")
	assertStringField(t, toolCalls[1], "id", "call-b")

	if _, hasContent := messages[1]["content"]; hasContent {
		t.Fatal("assistant tool-call message should not have content field")
	}

	assertStringField(t, messages[2], "role", "tool")
	assertStringField(t, messages[2], "tool_call_id", "call-a")
	assertStringField(t, messages[3], "role", "tool")
	assertStringField(t, messages[3], "tool_call_id", "call-b")
	assertStringField(t, messages[4], "role", "assistant")
	assertStringField(t, messages[4], "content", "done")
}

func TestHandleRouterReplayTrajectoryFallsBackToBodyParsing(t *testing.T) {
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	sessionID := "sess-fallback"

	requestBody := `{"messages":[{"role":"user","content":"hello"},{"role":"assistant","tool_calls":[{"id":"call-x","type":"function","function":{"name":"my_tool","arguments":"{}"}}]},{"role":"tool","content":"tool result","tool_call_id":"call-x"}]}`
	responseBody := `{"choices":[{"message":{"role":"assistant","content":"final"}}]}`

	_, err := recorder.AddRecord(routerreplay.RoutingRecord{
		ID:           "traj-fallback",
		RequestID:    sessionID,
		Timestamp:    time.Unix(1, 0).UTC(),
		RequestBody:  requestBody,
		ResponseBody: responseBody,
		// ToolTrace intentionally nil
	})
	if err != nil {
		t.Fatalf("failed to add record: %v", err)
	}

	router := &OpenAIRouter{
		ReplayRecorders: map[string]*routerreplay.Recorder{"default": recorder},
	}

	response := router.handleRouterReplayAPI("GET", "/v1/router_replay/trajectory?session_id="+sessionID)
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatal("expected immediate trajectory response")
	}

	body := decodeJSONBody(t, response.GetImmediateResponse().Body)
	assertIntField(t, body, "record_count", 1)

	messages := mustTrajectoryMessages(t, body)
	if len(messages) == 0 {
		t.Fatal("expected messages from body fallback, got none")
	}

	// The last message should be the assistant final response from the response body.
	last := messages[len(messages)-1]
	assertStringField(t, last, "role", "assistant")
	assertStringField(t, last, "content", "final")
}

func TestHandleRouterReplayTrajectoryReturnsEmptyMessagesForUnknownSession(t *testing.T) {
	router, _ := newTrajectoryTestRouter(t)

	response := router.handleRouterReplayAPI("GET", "/v1/router_replay/trajectory?session_id=unknown")
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatal("expected immediate trajectory response")
	}

	body := decodeJSONBody(t, response.GetImmediateResponse().Body)
	assertStringField(t, body, "object", "router_replay.trajectory")
	assertIntField(t, body, "record_count", 0)

	messages := mustTrajectoryMessages(t, body)
	if len(messages) != 0 {
		t.Fatalf("expected 0 messages for unknown session, got %d", len(messages))
	}
}

func TestHandleRouterReplayTrajectoryMethodNotAllowed(t *testing.T) {
	router, sessionID := newTrajectoryTestRouter(t)

	response := router.handleRouterReplayAPI("POST", "/v1/router_replay/trajectory?session_id="+sessionID)
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatal("expected immediate error response")
	}
	if got := response.GetImmediateResponse().GetStatus().GetCode(); got != typev3.StatusCode_MethodNotAllowed {
		t.Fatalf("expected 405 status, got %v", got)
	}
}

func TestHandleRouterReplayTrajectoryMissingSessionID(t *testing.T) {
	router, _ := newTrajectoryTestRouter(t)

	response := router.handleRouterReplayAPI("GET", "/v1/router_replay/trajectory")
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatal("expected immediate error response")
	}
	if got := response.GetImmediateResponse().GetStatus().GetCode(); got != typev3.StatusCode_BadRequest {
		t.Fatalf("expected 400 status, got %v", got)
	}

	body := decodeJSONBody(t, response.GetImmediateResponse().Body)
	errBody, ok := body["error"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected error payload, got %#v", body)
	}
	if got := errBody["message"]; got != "session_id is required" {
		t.Fatalf("unexpected error message: %#v", got)
	}
}

// --- helpers ---

func mustTrajectoryMessages(t *testing.T, body map[string]interface{}) []map[string]interface{} {
	t.Helper()
	raw, ok := body["messages"].([]interface{})
	if !ok {
		t.Fatalf("expected messages array, got %#v", body["messages"])
	}
	msgs := make([]map[string]interface{}, 0, len(raw))
	for _, item := range raw {
		msg, ok := item.(map[string]interface{})
		if !ok {
			t.Fatalf("expected message object, got %#v", item)
		}
		msgs = append(msgs, msg)
	}
	return msgs
}

func mustTrajectoryToolCalls(t *testing.T, msg map[string]interface{}) []map[string]interface{} {
	t.Helper()
	raw, ok := msg["tool_calls"].([]interface{})
	if !ok {
		t.Fatalf("expected tool_calls array, got %#v", msg["tool_calls"])
	}
	calls := make([]map[string]interface{}, 0, len(raw))
	for _, item := range raw {
		call, ok := item.(map[string]interface{})
		if !ok {
			t.Fatalf("expected tool call object, got %#v", item)
		}
		calls = append(calls, call)
	}
	return calls
}

func mustTrajectoryFunction(t *testing.T, toolCall map[string]interface{}) map[string]interface{} {
	t.Helper()
	fn, ok := toolCall["function"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected function object, got %#v", toolCall["function"])
	}
	return fn
}
