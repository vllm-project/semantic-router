package extproc

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func validityTestTools() []llmprotocol.Tool {
	return []llmprotocol.Tool{
		{
			Name:        "bash",
			InputSchema: json.RawMessage(`{"type":"object","required":["command","description"],"properties":{"command":{"type":"string"},"description":{"type":"string"}}}`),
		},
		{
			Name:        "freeform",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"note":{"type":"string"}}}`),
		},
	}
}

func assistantToolCallMessage(calls ...*llmprotocol.ToolCall) llmprotocol.Message {
	content := make([]llmprotocol.Content, 0, len(calls))
	for _, call := range calls {
		content = append(content, llmprotocol.Content{Kind: llmprotocol.ContentToolCall, ToolCall: call})
	}
	return llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: content}
}

func TestCountInvalidRecentToolCalls(t *testing.T) {
	tests := []struct {
		name     string
		tools    []llmprotocol.Tool
		messages []llmprotocol.Message
		want     int
	}{
		{
			name:  "missing required property",
			tools: validityTestTools(),
			messages: []llmprotocol.Message{
				{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "run tests"}}},
				assistantToolCallMessage(&llmprotocol.ToolCall{ID: "call_1", Name: "bash", Arguments: `{"command":"ls"}`}),
			},
			want: 1,
		},
		{
			name:  "all required properties present",
			tools: validityTestTools(),
			messages: []llmprotocol.Message{
				assistantToolCallMessage(&llmprotocol.ToolCall{ID: "call_1", Name: "bash", Arguments: `{"command":"ls","description":"list files"}`}),
			},
			want: 0,
		},
		{
			name:  "arguments are not JSON",
			tools: validityTestTools(),
			messages: []llmprotocol.Message{
				assistantToolCallMessage(&llmprotocol.ToolCall{ID: "call_1", Name: "bash", Arguments: `{"command":`}),
			},
			want: 1,
		},
		{
			name:  "tool not declared in request",
			tools: validityTestTools(),
			messages: []llmprotocol.Message{
				assistantToolCallMessage(&llmprotocol.ToolCall{ID: "call_1", Name: "unknown_tool", Arguments: "not json"}),
			},
			want: 0,
		},
		{
			name:  "custom tool call carries free-form input",
			tools: validityTestTools(),
			messages: []llmprotocol.Message{
				assistantToolCallMessage(&llmprotocol.ToolCall{ID: "call_1", Name: "bash", Kind: llmprotocol.ToolKind("custom"), Arguments: "not json"}),
			},
			want: 0,
		},
		{
			name:  "schema without required properties",
			tools: validityTestTools(),
			messages: []llmprotocol.Message{
				assistantToolCallMessage(&llmprotocol.ToolCall{ID: "call_1", Name: "freeform", Arguments: `{}`}),
			},
			want: 0,
		},
		{
			name:  "invalid call in an earlier assistant message only",
			tools: validityTestTools(),
			messages: []llmprotocol.Message{
				assistantToolCallMessage(&llmprotocol.ToolCall{ID: "call_1", Name: "bash", Arguments: `{"command":"ls"}`}),
				{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "again"}}},
				assistantToolCallMessage(&llmprotocol.ToolCall{ID: "call_2", Name: "bash", Arguments: `{"command":"pwd","description":"print directory"}`}),
			},
			want: 0,
		},
		{
			name:  "invalid call in an earlier item of the last assistant turn",
			tools: validityTestTools(),
			messages: []llmprotocol.Message{
				{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "run tests"}}},
				assistantToolCallMessage(&llmprotocol.ToolCall{ID: "call_1", Name: "bash", Arguments: `{"command":"ls"}`}),
				assistantToolCallMessage(&llmprotocol.ToolCall{ID: "call_2", Name: "bash", Arguments: `{"command":"ls","description":"list files"}`}),
			},
			want: 1,
		},
		{
			name:  "no assistant message",
			tools: validityTestTools(),
			messages: []llmprotocol.Message{
				{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "run tests"}}},
			},
			want: 0,
		},
		{
			name:  "multiple invalid calls in the last assistant message",
			tools: validityTestTools(),
			messages: []llmprotocol.Message{
				assistantToolCallMessage(
					&llmprotocol.ToolCall{ID: "call_1", Name: "bash", Arguments: `{"command":"ls"}`},
					&llmprotocol.ToolCall{ID: "call_2", Name: "bash", Arguments: `{"description":"list"}`},
				),
			},
			want: 2,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := &llmprotocol.Request{Model: "test-model", Tools: test.tools, Messages: test.messages}
			if got := countInvalidRecentToolCalls(request); got != test.want {
				t.Fatalf("countInvalidRecentToolCalls = %d, want %d", got, test.want)
			}
		})
	}
}

// A Responses input carries each function_call item as its own assistant
// message; the fact counts the whole turn, not only its last item.
func TestInvalidToolCallsCountsEveryItemOfAResponsesTurn(t *testing.T) {
	body := `{"model":"m","tools":[{"type":"function","name":"bash","parameters":{"type":"object","required":["command"],"properties":{"command":{"type":"string"}}}}],"input":[` +
		`{"role":"assistant","content":[{"type":"output_text","text":"calling"}]},` +
		`{"type":"function_call","call_id":"c1","name":"bash","arguments":"{}"},` +
		`{"type":"function_call","call_id":"c2","name":"bash","arguments":"{\"command\":\"ls\"}"}]}`
	request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIResponsesV1, []byte(body))
	if err != nil {
		t.Fatalf("decode request: %v", err)
	}
	if got := countInvalidRecentToolCalls(&request); got != 1 {
		t.Fatalf("countInvalidRecentToolCalls = %d, want 1", got)
	}
}

func TestExtractSemanticRequestSignalsInvalidToolCalls(t *testing.T) {
	request := &llmprotocol.Request{
		Model: "test-model",
		Tools: validityTestTools(),
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "run tests"}}},
			assistantToolCallMessage(&llmprotocol.ToolCall{ID: "call_1", Name: "bash", Arguments: `{"command":"ls"}`}),
		},
	}
	snapshot := extractSemanticRequestSignals(request)
	if snapshot.InvalidToolCalls != 1 {
		t.Fatalf("snapshot.InvalidToolCalls = %d, want 1", snapshot.InvalidToolCalls)
	}
}
