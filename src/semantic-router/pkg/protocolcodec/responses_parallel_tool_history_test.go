package protocolcodec

import (
	"bytes"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestResponsesParallelToolHistoryBecomesOneChatAssistantTurn(t *testing.T) {
	messages := projectResponsesHistoryToChat(t, `[
		{"role":"user","content":"use both results"},
		{"type":"function_call","id":"fc_a","call_id":"call_a","name":"weather","arguments":"{\"city\":\"Paris\"}"},
		{"type":"function_call","id":"fc_b","call_id":"call_b","name":"weather","arguments":"{\"city\":\"London\"}"},
		{"type":"function_call_output","call_id":"call_a","output":"sunny"},
		{"type":"function_call_output","call_id":"call_b","output":"rainy"}
	]`)

	if len(messages) != 4 {
		t.Fatalf("Chat message count = %d, want 4: %+v", len(messages), messages)
	}
	assistant := messages[1]
	if assistant.Role != "assistant" || len(assistant.ToolCalls) != 2 {
		t.Fatalf("parallel calls were not grouped into one assistant turn: %+v", assistant)
	}
	if assistant.ToolCalls[0].ID != "call_a" || assistant.ToolCalls[1].ID != "call_b" {
		t.Fatalf("parallel call order or IDs changed: %+v", assistant.ToolCalls)
	}
	if assistant.ToolCalls[0].Function.Name != "weather" || assistant.ToolCalls[0].Function.Arguments != `{"city":"Paris"}` ||
		assistant.ToolCalls[1].Function.Name != "weather" || assistant.ToolCalls[1].Function.Arguments != `{"city":"London"}` {
		t.Fatalf("call payload changed: %+v", assistant.ToolCalls)
	}
	if messages[2].Role != "tool" || messages[2].ToolCallID != "call_a" ||
		messages[3].Role != "tool" || messages[3].ToolCallID != "call_b" {
		t.Fatalf("tool results did not follow the grouped assistant turn: %+v", messages)
	}
}

func TestResponsesToolHistoryPreservesMessageBoundaries(t *testing.T) {
	for _, separator := range []string{
		`{"role":"user","content":"next"}`,
		`{"role":"assistant","content":[{"type":"output_text","text":"next"}]}`,
		`{"type":"reasoning","summary":[{"type":"summary_text","text":"next"}]}`,
	} {
		t.Run(separator, func(t *testing.T) {
			messages := projectResponsesHistoryToChat(t, `[
				{"type":"function_call","call_id":"a","name":"lookup","arguments":"{}"},`+separator+`,
				{"type":"function_call","call_id":"b","name":"lookup","arguments":"{}"},
				{"type":"function_call_output","call_id":"a","output":"one"},
				{"type":"function_call_output","call_id":"b","output":"two"}
			]`)
			// Projection must not repair non-adjacent histories by moving calls
			// across conversational content, even if a provider rejects that history.
			if len(messages) != 5 || len(messages[0].ToolCalls) != 1 || len(messages[2].ToolCalls) != 1 {
				t.Fatalf("merged across message boundary: %+v", messages)
			}
		})
	}
}

func TestResponsesGroupingDoesNotAcceptEmptyMessages(t *testing.T) {
	for _, item := range []string{`{"role":"assistant","content":[]}`, `{"type":"reasoning","summary":[]}`} {
		t.Run(item, func(t *testing.T) {
			_, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1,
				[]byte(`{"model":"m","input":[`+item+`]}`))
			if err == nil {
				t.Fatal("accepted an empty history message")
			}
		})
	}
}

func TestResponsesGroupingUsesNormalizedInstructions(t *testing.T) {
	for _, role := range []string{"system", "developer"} {
		t.Run(role, func(t *testing.T) {
			messages := projectResponsesHistoryToChat(t, `[
				{"type":"function_call","call_id":"a","name":"lookup","arguments":"{}"},
				{"role":"`+role+`","content":"Keep answers brief"},
				{"type":"function_call","call_id":"b","name":"lookup","arguments":"{}"},
				{"type":"function_call_output","call_id":"a","output":"one"},
				{"type":"function_call_output","call_id":"b","output":"two"}
			]`)
			if len(messages) != 4 || messages[0].Role != role || string(messages[0].Content) != `"Keep answers brief"` || len(messages[1].ToolCalls) != 2 {
				t.Fatalf("instruction normalization or grouping changed: %+v", messages)
			}
		})
	}
}

func TestResponsesParallelCustomAndMixedCalls(t *testing.T) {
	for _, first := range []string{
		`{"type":"function_call","call_id":"a","name":"lookup","arguments":"{}"}`,
		`{"type":"custom_tool_call","call_id":"a","name":"edit","input":"first patch"}`,
	} {
		t.Run(first, func(t *testing.T) {
			resultType := "function_call_output"
			if bytes.Contains([]byte(first), []byte("custom_tool_call")) {
				resultType = "custom_tool_call_output"
			}
			messages := projectResponsesHistoryToChat(t, `[`+first+`,
				{"type":"custom_tool_call","call_id":"b","name":"edit","input":"second patch"},
				{"type":"custom_tool_call","call_id":"c","name":"edit","input":"third patch"},
				{"type":"custom_tool_call_output","call_id":"c","output":"three"},
				{"type":"`+resultType+`","call_id":"a","output":"one"},
				{"type":"custom_tool_call_output","call_id":"b","output":"two"}
			]`)
			if len(messages) != 4 || len(messages[0].ToolCalls) != 3 {
				t.Fatalf("custom/mixed calls not grouped: %+v", messages)
			}
			calls := messages[0].ToolCalls
			if calls[0].ID != "a" || calls[1].ID != "b" || calls[2].ID != "c" ||
				calls[1].Type != "custom" || calls[1].Custom == nil || calls[1].Custom.Input != "second patch" ||
				calls[2].Custom == nil || calls[2].Custom.Input != "third patch" {
				t.Fatalf("call identity/payload changed: %+v", calls)
			}
			for i, id := range []string{"c", "a", "b"} {
				if messages[i+1].Role != "tool" || messages[i+1].ToolCallID != id {
					t.Fatalf("result order changed: %+v", messages)
				}
			}
		})
	}
}

func TestResponsesGroupingDoesNotMutateNativeHistory(t *testing.T) {
	engine := NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(`{"model":"m","input":[
		{"type":"function_call","id":"fc_a","call_id":"a","name":"lookup","arguments":"{}"},
		{"type":"function_call","id":"fc_b","call_id":"b","name":"lookup","arguments":"{}"},
		{"type":"function_call_output","call_id":"a","output":"one"},
		{"type":"function_call_output","call_id":"b","output":"two"}
	]}`))
	if err != nil {
		t.Fatal(err)
	}
	request.Generation++ // Exercise native re-encoding, not envelope replay.
	before, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	first, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	second, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	after, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first.Body, second.Body) || !bytes.Equal(before.Body, after.Body) || len(request.Messages) != 4 {
		t.Fatal("Chat projection mutated shared history or repeated encoding")
	}
	var native responsesRequestWire
	if err := json.Unmarshal(after.Body, &native); err != nil {
		t.Fatal(err)
	}
	var items []responsesItemWire
	if err := json.Unmarshal(native.Input, &items); err != nil {
		t.Fatal(err)
	}
	if len(items) != 4 || items[0].ID != "fc_a" || items[1].ID != "fc_b" || items[0].CallID != "a" || items[1].CallID != "b" {
		t.Fatalf("native item identities changed: %s", after.Body)
	}
}

func TestChatGroupingIsLimitedToResponsesSource(t *testing.T) {
	for _, source := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, ""} {
		t.Run(string(source), func(t *testing.T) {
			request := llmprotocol.Request{Trusted: llmprotocol.TrustedMetadata{SourceFormat: source}}
			for _, id := range []string{"a", "b"} {
				request.Messages = append(request.Messages, llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: id, Name: "lookup", Arguments: "{}"}}}})
			}
			var actual chatRequestWire
			if err := appendChatMessages(&actual, request); err != nil {
				t.Fatal(err)
			}
			var want []chatMessageWire
			for _, message := range request.Messages {
				encoded, err := encodeChatMessage(message)
				if err != nil {
					t.Fatal(err)
				}
				want = append(want, encoded)
			}
			if !reflect.DeepEqual(actual.Messages, want) {
				t.Fatalf("changed %s history: %+v", source, actual.Messages)
			}
		})
	}
}

func TestResponsesToolHistoryGroupingControls(t *testing.T) {
	t.Run("single call", func(t *testing.T) {
		messages := projectResponsesHistoryToChat(t, `[
			{"role":"user","content":"use the result"},
			{"type":"function_call","call_id":"call_a","name":"weather","arguments":"{}"},
			{"type":"function_call_output","call_id":"call_a","output":"sunny"}
		]`)
		if len(messages) != 3 || len(messages[1].ToolCalls) != 1 || messages[1].ToolCalls[0].ID != "call_a" {
			t.Fatalf("single-call history changed: %+v", messages)
		}
	})

	t.Run("sequential calls", func(t *testing.T) {
		messages := projectResponsesHistoryToChat(t, `[
			{"role":"user","content":"run two steps"},
			{"type":"function_call","call_id":"call_a","name":"first","arguments":"{}"},
			{"type":"function_call_output","call_id":"call_a","output":"done"},
			{"type":"function_call","call_id":"call_b","name":"second","arguments":"{}"},
			{"type":"function_call_output","call_id":"call_b","output":"done"}
		]`)
		if len(messages) != 5 {
			t.Fatalf("sequential Chat message count = %d, want 5: %+v", len(messages), messages)
		}
		if len(messages[1].ToolCalls) != 1 || messages[1].ToolCalls[0].ID != "call_a" ||
			messages[2].Role != "tool" || messages[2].ToolCallID != "call_a" ||
			len(messages[3].ToolCalls) != 1 || messages[3].ToolCalls[0].ID != "call_b" ||
			messages[4].Role != "tool" || messages[4].ToolCallID != "call_b" {
			t.Fatalf("calls were merged across an intervening tool result: %+v", messages)
		}
	})
}

func projectResponsesHistoryToChat(t *testing.T, input string) []chatMessageWire {
	t.Helper()
	engine := NewBuiltinEngine()
	body := []byte(`{"model":"source-model","input":` + input + `}`)
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	request.Model = "routed-model"
	request.Generation++
	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	var wire chatRequestWire
	if err := json.Unmarshal(encoded.Body, &wire); err != nil {
		t.Fatal(err)
	}
	return wire.Messages
}
