package extproc

import (
	"encoding/json"
	"strings"
	"testing"
)

// --- extractReasoningContentFromMessages tests ---

func TestExtractReasoningContent_MultiTurnWithReasoning(t *testing.T) {
	body := []byte(`{
		"model": "deepseek-v4-flash",
		"messages": [
			{"role": "system", "content": "You are helpful."},
			{"role": "user", "content": "What is 2+2?"},
			{"role": "assistant", "content": "4", "reasoning_content": "Let me think... 2+2=4"},
			{"role": "user", "content": "And 3+3?"},
			{"role": "assistant", "content": "6", "reasoning_content": "3+3=6"},
			{"role": "user", "content": "Thanks"}
		]
	}`)

	captured := extractReasoningContentFromMessages(body)
	if len(captured) != 2 {
		t.Fatalf("expected 2 captured entries, got %d", len(captured))
	}
	// Index 2 = first assistant message
	if string(captured[2]) != `"Let me think... 2+2=4"` {
		t.Errorf("index 2: got %s", captured[2])
	}
	// Index 4 = second assistant message
	if string(captured[4]) != `"3+3=6"` {
		t.Errorf("index 4: got %s", captured[4])
	}
}

func TestExtractReasoningContent_EmptyString(t *testing.T) {
	// DeepSeek requires empty strings to be round-tripped.
	body := []byte(`{
		"messages": [
			{"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": [{"id":"tc1","type":"function","function":{"name":"search","arguments":"{}"}}]},
			{"role": "tool", "content": "result", "tool_call_id": "tc1"}
		]
	}`)

	captured := extractReasoningContentFromMessages(body)
	if len(captured) != 1 {
		t.Fatalf("expected 1 captured entry, got %d", len(captured))
	}
	if string(captured[0]) != `""` {
		t.Errorf("expected empty string JSON, got %s", captured[0])
	}
}

func TestExtractReasoningContent_NoReasoningContent(t *testing.T) {
	body := []byte(`{
		"messages": [
			{"role": "user", "content": "hello"},
			{"role": "assistant", "content": "hi"}
		]
	}`)

	captured := extractReasoningContentFromMessages(body)
	if captured != nil {
		t.Fatalf("expected nil map for request without reasoning_content, got %v", captured)
	}
}

func TestExtractReasoningContent_NoMessages(t *testing.T) {
	body := []byte(`{"model": "gpt-4"}`)

	captured := extractReasoningContentFromMessages(body)
	if captured != nil {
		t.Fatalf("expected nil map for request without messages, got %v", captured)
	}
}

func TestExtractReasoningContent_OnlyNonAssistantRoles(t *testing.T) {
	// Even if a non-standard client somehow puts reasoning_content on a user
	// message, we only capture from assistant messages.
	body := []byte(`{
		"messages": [
			{"role": "user", "content": "hello", "reasoning_content": "should be ignored"}
		]
	}`)

	captured := extractReasoningContentFromMessages(body)
	if captured != nil {
		t.Fatalf("expected nil map for non-assistant reasoning_content, got %v", captured)
	}
}

// --- restoreReasoningContentToMessages tests ---

func TestRestoreReasoningContent_RoundTrip(t *testing.T) {
	// Simulate: original body has reasoning_content, SDK strips it, we restore.
	original := []byte(`{
		"model": "deepseek-v4-flash",
		"messages": [
			{"role": "user", "content": "hi"},
			{"role": "assistant", "content": "hello", "reasoning_content": "thinking..."},
			{"role": "user", "content": "thanks"}
		]
	}`)

	captured := extractReasoningContentFromMessages(original)
	if len(captured) != 1 {
		t.Fatalf("expected 1 captured entry, got %d", len(captured))
	}

	// Simulate SDK round-trip: reasoning_content is gone.
	sdkOutput := []byte(`{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"},{"role":"user","content":"thanks"}]}`)

	restored, err := restoreReasoningContentToMessages(sdkOutput, captured)
	if err != nil {
		t.Fatalf("restore error: %v", err)
	}

	// Verify reasoning_content is back.
	var parsed struct {
		Messages []struct {
			Role             string `json:"role"`
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(restored, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if len(parsed.Messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(parsed.Messages))
	}
	if parsed.Messages[1].ReasoningContent != "thinking..." {
		t.Errorf("expected reasoning_content 'thinking...', got %q", parsed.Messages[1].ReasoningContent)
	}
}

func TestRestoreReasoningContent_EmptyStringPreserved(t *testing.T) {
	captured := map[int]json.RawMessage{
		0: json.RawMessage(`""`),
	}

	body := []byte(`{"messages":[{"role":"assistant","content":""}]}`)
	restored, err := restoreReasoningContentToMessages(body, captured)
	if err != nil {
		t.Fatalf("restore error: %v", err)
	}

	// Verify the field exists with empty string value.
	if !strings.Contains(string(restored), `"reasoning_content":""`) {
		t.Errorf("expected empty reasoning_content in output, got: %s", restored)
	}
}

func TestRestoreReasoningContent_NilCapturedNoOp(t *testing.T) {
	body := []byte(`{"messages":[{"role":"user","content":"hello"}]}`)

	restored, err := restoreReasoningContentToMessages(body, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if string(restored) != string(body) {
		t.Error("expected body to be unchanged when captured is nil")
	}
}

func TestRestoreReasoningContent_EmptyCapturedNoOp(t *testing.T) {
	body := []byte(`{"messages":[{"role":"user","content":"hello"}]}`)

	restored, err := restoreReasoningContentToMessages(body, map[int]json.RawMessage{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if string(restored) != string(body) {
		t.Error("expected body to be unchanged when captured is empty")
	}
}

func TestReasoningContentPassthrough_FullPipeline(t *testing.T) {
	// Simulates the complete flow: 15 assistant messages with reasoning_content
	// (matching the bug report's scenario of a 10+ turn agent conversation).
	body := []byte(`{
		"model": "vllm-sr/auto",
		"messages": [
			{"role": "system", "content": "You are an agent."},
			{"role": "user", "content": "Search for flights."},
			{"role": "assistant", "content": "", "reasoning_content": "I need to search for flights. Let me use the search tool.", "tool_calls": [{"id":"tc1","type":"function","function":{"name":"search_flights","arguments":"{\"dest\":\"NYC\"}"}}]},
			{"role": "tool", "content": "{\"flights\":[{\"id\":1}]}", "tool_call_id": "tc1"},
			{"role": "assistant", "content": "I found a flight.", "reasoning_content": "The search returned one flight to NYC."},
			{"role": "user", "content": "Book it."},
			{"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": [{"id":"tc2","type":"function","function":{"name":"book_flight","arguments":"{\"id\":1}"}}]},
			{"role": "tool", "content": "{\"status\":\"booked\"}", "tool_call_id": "tc2"},
			{"role": "assistant", "content": "Booked!", "reasoning_content": "Booking confirmed."}
		],
		"tools": [{"type":"function","function":{"name":"search_flights","parameters":{}}}],
		"max_completion_tokens": 8192
	}`)

	// Step 1: Extract before SDK parse.
	captured := extractReasoningContentFromMessages(body)

	// We have 4 assistant messages at indices 2, 4, 6, 8.
	if len(captured) != 4 {
		t.Fatalf("expected 4 captured entries, got %d: %v", len(captured), captured)
	}

	// Step 2: Simulate SDK round-trip stripping reasoning_content.
	stripped := []byte(`{"model":"deepseek-v4-flash","messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"Search for flights."},{"role":"assistant","content":"","tool_calls":[{"id":"tc1","type":"function","function":{"name":"search_flights","arguments":"{\"dest\":\"NYC\"}"}}]},{"role":"tool","content":"{\"flights\":[{\"id\":1}]}","tool_call_id":"tc1"},{"role":"assistant","content":"I found a flight."},{"role":"user","content":"Book it."},{"role":"assistant","content":"","tool_calls":[{"id":"tc2","type":"function","function":{"name":"book_flight","arguments":"{\"id\":1}"}}]},{"role":"tool","content":"{\"status\":\"booked\"}","tool_call_id":"tc2"},{"role":"assistant","content":"Booked!"}],"tools":[{"type":"function","function":{"name":"search_flights","parameters":{}}}],"max_completion_tokens":8192}`)

	// Step 3: Restore.
	restored, err := restoreReasoningContentToMessages(stripped, captured)
	if err != nil {
		t.Fatalf("restore error: %v", err)
	}

	// Step 4: Verify all 4 assistant messages have their reasoning_content back.
	var parsed struct {
		Messages []struct {
			Role             string `json:"role"`
			ReasoningContent *string `json:"reasoning_content,omitempty"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(restored, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	expectedRC := map[int]string{
		2: "I need to search for flights. Let me use the search tool.",
		4: "The search returned one flight to NYC.",
		6: "",
		8: "Booking confirmed.",
	}

	for idx, want := range expectedRC {
		msg := parsed.Messages[idx]
		if msg.Role != "assistant" {
			t.Errorf("messages[%d]: expected role assistant, got %s", idx, msg.Role)
			continue
		}
		if msg.ReasoningContent == nil {
			t.Errorf("messages[%d]: reasoning_content is nil, expected %q", idx, want)
			continue
		}
		if *msg.ReasoningContent != want {
			t.Errorf("messages[%d]: reasoning_content = %q, want %q", idx, *msg.ReasoningContent, want)
		}
	}

	// Non-assistant messages must NOT have reasoning_content.
	for i, msg := range parsed.Messages {
		if msg.Role != "assistant" && msg.ReasoningContent != nil {
			t.Errorf("messages[%d] (role=%s): should not have reasoning_content", i, msg.Role)
		}
	}
}
