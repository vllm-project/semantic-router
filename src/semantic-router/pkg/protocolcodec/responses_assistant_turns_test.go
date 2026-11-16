package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// One Codex step as Responses items: reasoning, a message and two calls.
const codexStepHistory = `[
	{"role":"user","content":"find the lost commit"},
	{"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"check the reflog"}]},
	{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Let me look at the reflog."}]},
	{"type":"function_call","call_id":"a","name":"exec_command","arguments":"{\"cmd\":\"git reflog\"}"},
	{"type":"function_call","call_id":"b","name":"exec_command","arguments":"{\"cmd\":\"git status\"}"},
	{"type":"function_call_output","call_id":"a","output":"one"},
	{"type":"function_call_output","call_id":"b","output":"two"}
]`

func TestResponsesStepBecomesOneChatAssistantTurn(t *testing.T) {
	messages := projectResponsesHistoryToChat(t, codexStepHistory)
	if len(messages) != 4 {
		t.Fatalf("Chat message count = %d, want 4: %+v", len(messages), messages)
	}
	assistant := messages[1]
	if assistant.Role != "assistant" || assistant.Reasoning == nil || string(assistant.Content) != `"Let me look at the reflog."` ||
		len(assistant.ToolCalls) != 2 || assistant.ToolCalls[0].ID != "a" || assistant.ToolCalls[1].ID != "b" {
		t.Fatalf("the step was not one assistant turn: %+v", assistant)
	}
}

func TestResponsesStepBecomesOneMessagesAssistantTurn(t *testing.T) {
	engine := NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1,
		[]byte(`{"model":"m","input":`+codexStepHistory+`}`))
	if err != nil {
		t.Fatal(err)
	}
	request.Generation++
	before := len(request.Messages)
	encoded, err := engine.EncodeRequest(llmprotocol.AnthropicMessagesV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	if len(request.Messages) != before {
		t.Fatalf("projection changed the shared history: %d messages, want %d", len(request.Messages), before)
	}
	var wire struct {
		Messages []struct {
			Role    string `json:"role"`
			Content []struct {
				Type string `json:"type"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(encoded.Body, &wire); err != nil {
		t.Fatal(err)
	}
	var roles, blocks []string
	for _, message := range wire.Messages {
		roles = append(roles, message.Role)
	}
	for _, block := range wire.Messages[1].Content {
		blocks = append(blocks, block.Type)
	}
	if len(wire.Messages) < 3 || wire.Messages[1].Role != "assistant" || wire.Messages[2].Role != "user" ||
		len(blocks) != 4 || blocks[0] != "thinking" || blocks[1] != "text" || blocks[2] != "tool_use" || blocks[3] != "tool_use" {
		t.Fatalf("the step was not one assistant turn: roles %v, assistant blocks %v", roles, blocks)
	}
}
