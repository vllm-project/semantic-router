package protocolcodec

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// cutChatReply is a buffered Chat reply whose output limit cut its second call
// inside the arguments.
const cutChatReply = `{"id":"chatcmpl-cut","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,` +
	`"message":{"role":"assistant","content":null,"tool_calls":[` +
	`{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}},` +
	`{"id":"call_2","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Lyon\",\"unit\":\"cel"}}]},` +
	`"finish_reason":"length"}],"usage":{"prompt_tokens":10,"completion_tokens":24,"total_tokens":34}}`

// A buffered reply that the output limit cut inside a call reaches every
// client with the limit stop. Chat and Responses clients get the arguments as
// the backend sent them; tool_use.input must be an object, so a Messages client
// gets the members completed before the cut, reported as approximated.
func TestBufferedToolCallCutByTheOutputLimitTranslates(t *testing.T) {
	engine := NewBuiltinEngine()
	tests := []struct {
		target    llmprotocol.WireFormat
		fragments []string
		input     string
	}{
		{llmprotocol.OpenAIChatV1, []string{`"finish_reason":"length"`, `"arguments":"{\"city\":\"Lyon\",\"unit\":\"cel"`}, ""},
		{llmprotocol.OpenAIResponsesV1, []string{`"status":"incomplete"`, `"incomplete_details":{"reason":"max_output_tokens"}`, `"arguments":"{\"city\":\"Lyon\",\"unit\":\"cel"`}, ""},
		{llmprotocol.AnthropicMessagesV1, []string{`"stop_reason":"max_tokens"`}, `{"city":"Lyon"}`},
	}
	for _, test := range tests {
		t.Run(string(test.target), func(t *testing.T) {
			result, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, test.target, []byte(cutChatReply), func(response *llmprotocol.Response) error {
				response.Model = "routed-model"
				return nil
			})
			if err != nil {
				t.Fatalf("translate: %v", err)
			}
			for _, fragment := range test.fragments {
				if !strings.Contains(string(result.Body), fragment) {
					t.Errorf("want %s in:\n%s", fragment, result.Body)
				}
			}
			approximated := slices.ContainsFunc(result.Diagnostics, func(diagnostic llmprotocol.Diagnostic) bool {
				return diagnostic.Field == "content.tool_use.input" && diagnostic.Action == llmprotocol.DiagnosticApproximated
			})
			if approximated != (test.input != "") {
				t.Errorf("approximated tool_use.input diagnostic = %t, want %t: %+v", approximated, test.input != "", result.Diagnostics)
			}
			if test.input == "" {
				return
			}
			var message struct {
				Content []struct {
					ID    string          `json:"id"`
					Input json.RawMessage `json:"input"`
				} `json:"content"`
			}
			if err = json.Unmarshal(result.Body, &message); err != nil {
				t.Fatal(err)
			}
			inputs := map[string]string{}
			for _, block := range message.Content {
				inputs[block.ID] = string(block.Input)
			}
			if inputs["call_1"] != `{"city":"Paris"}` || inputs["call_2"] != test.input {
				t.Fatalf("tool_use inputs = %v, want call_1 whole and call_2 %s", inputs, test.input)
			}
		})
	}
}

// vLLM's last chunk can carry a call's latest fragment and its tool parser's
// flush of the rest as two entries with one index, and a chunk can carry a
// call's name and its first fragment the same way. They continue one call, in
// order, as clients join them. Entries with one index that name different
// calls still fail.
func TestChatStreamToolEntriesWithOneIndexContinueOneCall(t *testing.T) {
	chunk := func(toolCalls, finish string) string {
		return `data: {"id":"chatcmpl-split","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,` +
			`"delta":{"tool_calls":[` + toolCalls + `]},"logprobs":null,"finish_reason":` + finish + `}]}` + "\n\n"
	}
	named := `{"id":"call_1","type":"function","index":0,"function":{"name":"get_weather"}}`
	fragment := func(arguments string) string {
		encoded, _ := json.Marshal(arguments)
		return `{"index":0,"function":{"arguments":` + string(encoded) + `}}`
	}
	tests := map[string]struct {
		frames []string
		code   string
	}{
		"name and first fragment": {frames: []string{chunk(named+","+fragment(`{"city": "`), "null"), chunk(fragment(`Paris"}`), `"tool_calls"`)}},
		"fragment and flush":      {frames: []string{chunk(named, "null"), chunk(fragment(`{"city": "Par`)+","+fragment(`is"}`), `"length"`)}},
		"the same ID twice": {frames: []string{
			chunk(named+`,{"id":"call_1","index":0,"function":{"arguments":"{\"city\": \"Paris\"}"}}`, `"tool_calls"`),
		}},
		"two IDs": {code: "duplicate_stream_tool_index", frames: []string{
			chunk(named+`,{"id":"call_2","type":"function","index":0,"function":{"name":"get_weather"}}`, "null"),
		}},
		"two names": {code: "duplicate_stream_tool_index", frames: []string{
			chunk(named+`,{"index":0,"function":{"name":"get_time"}}`, "null"),
		}},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			decoder := OpenAIChatCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy())
			var events []llmprotocol.Event
			var err error
			for _, frame := range append(test.frames, "data: [DONE]\n\n") {
				var decoded []llmprotocol.Event
				decoded, _, err = decoder.Push([]byte(frame))
				events = append(events, decoded...)
				if err != nil {
					break
				}
			}
			if test.code != "" {
				requireProtocolErrorCode(t, err, test.code)
				return
			}
			if err != nil {
				t.Fatalf("push: %v", err)
			}
			var completed *llmprotocol.ToolCall
			for _, event := range events {
				if event.Type == llmprotocol.EventOutputItemCompleted && event.ToolCall != nil {
					completed = event.ToolCall
				}
			}
			if completed == nil || completed.ID != "call_1" || completed.Name != "get_weather" || completed.Arguments != `{"city": "Paris"}` {
				t.Fatalf("completed call = %+v, want call_1 get_weather {\"city\": \"Paris\"}", completed)
			}
		})
	}
}
