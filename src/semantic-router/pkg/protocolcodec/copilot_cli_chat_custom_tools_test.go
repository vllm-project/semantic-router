package protocolcodec

import (
	"encoding/json"
	"errors"
	"os"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The two /v1/chat/completions requests Copilot CLI 1.0.88 sends for one
// apply_patch call when COPILOT_PROVIDER_MODEL_ID names a GPT model. The
// system prompt and descriptions are shortened; IDs are placeholders.
const copilotToolLoopFixture = "testdata/clients/copilot-cli-1.0.88-gpt-tool-loop.json"

type copilotChatBody struct {
	Tools    []json.RawMessage `json:"tools"`
	Messages []struct {
		Role       string            `json:"role"`
		ToolCallID string            `json:"tool_call_id"`
		ToolCalls  []json.RawMessage `json:"tool_calls"`
	} `json:"messages"`
}

func loadCopilotToolLoop(t *testing.T) []json.RawMessage {
	t.Helper()
	raw, err := os.ReadFile(copilotToolLoopFixture)
	if err != nil {
		t.Fatal(err)
	}
	var turns []json.RawMessage
	if err := json.Unmarshal(raw, &turns); err != nil {
		t.Fatal(err)
	}
	if len(turns) != 2 {
		t.Fatalf("fixture has %d turns, want 2", len(turns))
	}
	return turns
}

func TestCopilotToolLoopReachesOpenAIBackendsWithItsCustomTool(t *testing.T) {
	engine := NewBuiltinEngine()
	for turn, body := range loadCopilotToolLoop(t) {
		request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, body)
		if err != nil {
			t.Fatalf("turn %d: Copilot request rejected: %v", turn+1, err)
		}
		request.Model = "routed-model"
		request.Generation++

		encoded, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, envelope)
		if err != nil {
			t.Fatalf("turn %d to Chat: %v", turn+1, err)
		}
		var sent, dispatched copilotChatBody
		if err := json.Unmarshal(body, &sent); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(encoded.Body, &dispatched); err != nil {
			t.Fatal(err)
		}
		if !containsJSON(dispatched.Tools, copilotCustomTool(t, sent.Tools)) {
			t.Fatalf("turn %d: Chat dispatch changed or lost the custom tool", turn+1)
		}

		responses, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, request, envelope)
		if err != nil || !json.Valid(responses.Body) {
			t.Fatalf("turn %d to Responses returned %v: %s", turn+1, err, responses.Body)
		}
		_, err = engine.EncodeRequest(llmprotocol.AnthropicMessagesV1, request, envelope)
		var protocolError *llmprotocol.ProtocolError
		if !errors.As(err, &protocolError) || protocolError.Category != llmprotocol.ErrorUnsupportedFeature ||
			protocolError.Code != "unsupported_capability" {
			t.Fatalf("turn %d to Anthropic returned %v, want unsupported_capability", turn+1, err)
		}
	}
}

func TestCopilotToolResultTurnKeepsTheCustomCall(t *testing.T) {
	engine := NewBuiltinEngine()
	body := loadCopilotToolLoop(t)[1]
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, body)
	if err != nil {
		t.Fatal(err)
	}
	request.Model = "routed-model"
	request.Generation++
	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}

	var sent, chat copilotChatBody
	if err := json.Unmarshal(body, &sent); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(encoded.Body, &chat); err != nil {
		t.Fatal(err)
	}
	sentCall := sent.Messages[len(sent.Messages)-2].ToolCalls[0]
	last := len(chat.Messages) - 1
	if call := chat.Messages[last-1]; len(call.ToolCalls) != 1 || !jsonSemanticallyEqual(call.ToolCalls[0], sentCall) {
		t.Fatalf("custom tool call did not reach Chat unchanged: %s", call.ToolCalls)
	}
	if result := chat.Messages[last]; result.Role != "tool" || result.ToolCallID != "call_1" {
		t.Fatalf("tool result did not reach Chat: %+v", result)
	}
}

func copilotCustomTool(t *testing.T, tools []json.RawMessage) json.RawMessage {
	t.Helper()
	for _, tool := range tools {
		var probe struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(tool, &probe) == nil && probe.Type == "custom" {
			return tool
		}
	}
	t.Fatal("fixture has no custom tool")
	return nil
}

func containsJSON(values []json.RawMessage, want json.RawMessage) bool {
	for _, value := range values {
		if jsonSemanticallyEqual(value, want) {
			return true
		}
	}
	return false
}
