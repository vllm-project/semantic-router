package handlers

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestResolveOpenClawSessionFileFromIndex(t *testing.T) {
	indexData := []byte(`{
	  "agent:vllm-sr:openai-user:room-a:worker-a": {
	    "sessionFile": "/state/agents/vllm-sr/sessions/abc.jsonl"
	  }
	}`)

	sessionFile, ok := resolveOpenClawSessionFileFromIndex(indexData, []string{
		"agent:vllm-sr:openresponses-user:room-a:worker-a",
		"agent:vllm-sr:openai-user:room-a:worker-a",
	})
	if !ok {
		t.Fatal("expected session file to resolve")
	}
	if sessionFile != "/state/agents/vllm-sr/sessions/abc.jsonl" {
		t.Fatalf("sessionFile = %q", sessionFile)
	}
}

func TestParseOpenClawSessionToolTraceLine(t *testing.T) {
	toolCallLine := `{"type":"message","message":{"role":"assistant","content":[{"type":"toolCall","id":"call_1","name":"exec","arguments":{"command":"pwd"}}]}}`
	toolResultLine := `{"type":"message","message":{"role":"toolResult","toolCallId":"call_1","toolName":"exec","content":[{"type":"text","text":"/workspace"}],"isError":false}}`

	steps := make(map[string]openClawSessionToolStep)
	order := make([]string, 0)

	order, changed := parseOpenClawSessionToolTraceLine(toolCallLine, steps, order)
	if !changed {
		t.Fatal("expected tool call line to change state")
	}
	if steps["call_1"].Status != "running" {
		t.Fatalf("tool call status = %q, want running", steps["call_1"].Status)
	}

	order, changed = parseOpenClawSessionToolTraceLine(toolResultLine, steps, order)
	if !changed {
		t.Fatal("expected tool result line to change state")
	}
	if steps["call_1"].Status != "completed" {
		t.Fatalf("tool result status = %q, want completed", steps["call_1"].Status)
	}
	if steps["call_1"].Result != "/workspace" {
		t.Fatalf("tool result = %q", steps["call_1"].Result)
	}
	if len(order) != 1 || order[0] != "call_1" {
		t.Fatalf("order = %#v", order)
	}
}

func TestOpenClawSessionToolTraceInitialOffset(t *testing.T) {
	if got := openClawSessionToolTraceInitialOffset(128, 128); got != 128 {
		t.Fatalf("offset = %d, want 128", got)
	}
	if got := openClawSessionToolTraceInitialOffset(128, 64); got != 0 {
		t.Fatalf("offset = %d, want 0 for truncated baseline", got)
	}
	if got := openClawSessionToolTraceInitialOffset(0, 64); got != 0 {
		t.Fatalf("offset = %d, want 0 for missing baseline", got)
	}
}

func TestEncodeRoomMessageToolTraceMetadata(t *testing.T) {
	steps := []openClawSessionToolStep{
		{ID: "call_1", Name: "exec", Status: "completed", Result: "/workspace"},
		{ID: "call_2", Name: "read", Status: "running"},
	}
	encoded, err := encodeRoomMessageToolTraceMetadata(steps)
	if err != nil {
		t.Fatalf("encodeRoomMessageToolTraceMetadata() error = %v", err)
	}

	message := &ClawRoomMessage{ID: "msg-1"}
	attachRoomMessageToolTraceMetadata(message, steps)
	if message.Metadata[roomMessageToolTraceMetadataKey] != encoded {
		t.Fatalf("metadata = %q, want %q", message.Metadata[roomMessageToolTraceMetadataKey], encoded)
	}

	var decoded []openClawSessionToolStep
	if err := json.Unmarshal([]byte(encoded), &decoded); err != nil {
		t.Fatalf("decode metadata: %v", err)
	}
	if len(decoded) != 2 || decoded[0].ID != "call_1" || decoded[1].ID != "call_2" {
		t.Fatalf("decoded order = %#v", decoded)
	}
}

func TestSplitOpenClawSessionToolTraceLines(t *testing.T) {
	toolCallLine := `{"type":"message","message":{"role":"assistant","content":[{"type":"toolCall","id":"call_1","name":"exec","arguments":{"command":"pwd"}}]}}`
	largeLine := strings.Repeat("x", openClawSessionToolTraceMaxLineBytes/2) + toolCallLine

	lines, err := splitOpenClawSessionToolTraceLines([]byte(largeLine + "\n" + toolCallLine))
	if err != nil {
		t.Fatalf("splitOpenClawSessionToolTraceLines() error = %v", err)
	}
	if len(lines) != 2 {
		t.Fatalf("lines = %d, want 2", len(lines))
	}
}

func TestOpenClawChatCompletionsSessionKeys(t *testing.T) {
	keys := openClawChatCompletionsSessionKeys("room-a:worker-a")
	if len(keys) != 2 {
		t.Fatalf("keys = %#v", keys)
	}
	if keys[0] != "agent:vllm-sr:openai-user:room-a:worker-a" {
		t.Fatalf("first key = %q", keys[0])
	}
}
