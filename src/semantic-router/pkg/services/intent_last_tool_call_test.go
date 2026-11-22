package services

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Routing Preview reads the latest assistant turn's calls, names and raw
// arguments, as live routing does, so a last_tool_call rule matches the same
// history in both.
func TestIntentRequestResolveSignalInput_ReadsTheLastToolCall(t *testing.T) {
	call := func(id, name, args string) json.RawMessage {
		return mustMessageContent(t, map[string]any{
			"id": id, "type": "function",
			"function": map[string]string{"name": name, "arguments": args},
		})
	}
	req := IntentRequest{Messages: []IntentMessage{
		{Role: "user", Content: mustMessageContent(t, "Read the notes.")},
		{Role: "assistant", ToolCalls: []json.RawMessage{call("call_1", "ls", `{}`)}},
		{Role: "tool", ToolCallID: "call_1", Content: mustMessageContent(t, "notes.txt")},
		{Role: "assistant", ToolCalls: []json.RawMessage{call("call_2", "exec_command", `{"cmd":"cat notes.txt"}`)}},
		{Role: "tool", ToolCallID: "call_2", Content: mustMessageContent(t, "hello")},
		{Role: "user", Content: mustMessageContent(t, "Thanks.")},
	}}

	input, err := req.resolveSignalInput()
	require.NoError(t, err)

	assert.Equal(t, []string{"exec_command"}, input.conversationFacts.LastAssistantToolNames)
	assert.Equal(t, []string{`{"cmd":"cat notes.txt"}`}, input.conversationFacts.LastAssistantToolArgs)
}
