package protocolcodec

import (
	"bytes"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestChatEmptyToolResultContent(t *testing.T) {
	fixtures := map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIChatV1:        `{"model":"alias","messages":[{"role":"user","content":"lookup"},{"role":"assistant","tool_calls":[{"id":"call_a","type":"function","function":{"name":"lookup","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call_a","content":%s}]}`,
		llmprotocol.OpenAIResponsesV1:   `{"model":"alias","input":[{"role":"user","content":"lookup"},{"type":"function_call","call_id":"call_a","name":"lookup","arguments":"{}"},{"type":"function_call_output","call_id":"call_a","output":%s}]}`,
		llmprotocol.AnthropicMessagesV1: `{"model":"alias","max_tokens":32,"messages":[{"role":"user","content":"lookup"},{"role":"assistant","content":[{"type":"tool_use","id":"call_a","name":"lookup","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_a","content":%s}]}]}`,
	}
	for source, fixture := range fixtures {
		for _, value := range []struct{ name, wire, want string }{
			{"empty array", `[]`, `""`},
			{"empty string", `""`, `""`},
			{"text", `"result"`, `"result"`},
		} {
			t.Run(string(source)+"/"+value.name, func(t *testing.T) {
				input := []byte(fmt.Sprintf(fixture, value.wire))
				original := bytes.Clone(input)
				if source == llmprotocol.OpenAIChatV1 {
					passthrough, passthroughErr := NewBuiltinEngine().TranslateRequest(source, source, input, nil)
					if passthroughErr != nil || !bytes.Equal(passthrough.Body, input) {
						t.Fatalf("native Chat passthrough changed: error=%v body=%s", passthroughErr, passthrough.Body)
					}
				}
				// Model resolution must exercise encoding even for Chat-to-Chat.
				result, err := NewBuiltinEngine().TranslateRequest(source, llmprotocol.OpenAIChatV1, input, func(request *llmprotocol.Request) error {
					request.Model = "provider-model"
					return nil
				})
				if err != nil || len(result.Diagnostics) != 0 {
					t.Fatalf("translation: error=%v diagnostics=%+v", err, result.Diagnostics)
				}
				if !bytes.Equal(input, original) {
					t.Fatal("mutated source bytes")
				}
				var wire struct {
					Model    string `json:"model"`
					Messages []struct {
						Role    string          `json:"role"`
						CallID  string          `json:"tool_call_id"`
						Content json.RawMessage `json:"content"`
					} `json:"messages"`
				}
				if err := json.Unmarshal(result.Body, &wire); err != nil {
					t.Fatal(err)
				}
				if wire.Model != "provider-model" || len(wire.Messages) != 3 {
					t.Fatalf("unexpected history: %s", result.Body)
				}
				tool := wire.Messages[2]
				if tool.Role != "tool" || tool.CallID != "call_a" || string(tool.Content) != value.want {
					t.Fatalf("tool result must retain explicit content and call ID: %s", result.Body)
				}
				if len(wire.Messages[1].Content) != 0 {
					t.Fatalf("assistant content unexpectedly changed: %s", result.Body)
				}
			})
		}
	}
}
