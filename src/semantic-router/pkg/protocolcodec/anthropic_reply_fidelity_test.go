package protocolcodec

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// messagesStreamEvent is the part of a Messages stream event these tests read.
type messagesStreamEvent struct {
	Type    string `json:"type"`
	Index   int    `json:"index"`
	Message struct {
		ID    string `json:"id"`
		Usage struct {
			InputTokens int64 `json:"input_tokens"`
		} `json:"usage"`
	} `json:"message"`
	ContentBlock map[string]json.RawMessage `json:"content_block"`
	Delta        map[string]json.RawMessage `json:"delta"`
	Usage        struct {
		InputTokens int64 `json:"input_tokens"`
	} `json:"usage"`
}

func streamToMessagesClient(t *testing.T, source llmprotocol.WireFormat, estimate int64, payload []byte) []messagesStreamEvent {
	t.Helper()
	stream, err := NewBuiltinEngine().NewStream(source, llmprotocol.AnthropicMessagesV1, llmprotocol.StreamContext{
		Context: t.Context(), PublicModel: "public-model", EstimatedInputTokens: estimate,
	})
	if err != nil {
		t.Fatal(err)
	}
	frames, _, _, err := stream.Push(payload)
	if err != nil {
		t.Fatal(err)
	}
	final, _, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	var events []messagesStreamEvent
	for _, frame := range append(frames, final...) {
		parsed, err := parseSSEFrame(frame, len(frame))
		if err != nil {
			t.Fatal(err)
		}
		var event messagesStreamEvent
		if err := json.Unmarshal(parsed.Data, &event); err != nil {
			t.Fatalf("frame %q: %v", frame, err)
		}
		events = append(events, event)
	}
	return events
}

// blockEvents describes, in order, the events of one content block.
func blockEvents(events []messagesStreamEvent, index int) []string {
	var described []string
	for _, event := range events {
		if event.Index != index {
			continue
		}
		switch event.Type {
		case "content_block_start":
			description := "start " + jsonText(event.ContentBlock["type"])
			if signature, ok := event.ContentBlock["signature"]; ok {
				description += " signature=" + string(signature)
			}
			described = append(described, description)
		case "content_block_delta":
			description := jsonText(event.Delta["type"])
			if signature, ok := event.Delta["signature"]; ok {
				description += " " + string(signature)
			}
			described = append(described, description)
		case "content_block_stop":
			described = append(described, "stop")
		}
	}
	return described
}

func jsonText(raw json.RawMessage) string {
	var text string
	_ = json.Unmarshal(raw, &text)
	return text
}

const chatReasoningReply = `{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,` +
	`"message":{"role":"assistant","reasoning_content":"check","content":"answer"},"finish_reason":"stop"}],` +
	`"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`

func chatReasoningStream() []byte {
	chunk := func(delta, finish string) string {
		return `data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":` +
			delta + `,"finish_reason":` + finish + `}]}` + "\n\n"
	}
	return []byte(chunk(`{"role":"assistant","reasoning_content":"check"}`, "null") +
		chunk(`{"content":"answer"}`, "null") + chunk(`{}`, `"stop"`) + "data: [DONE]\n\n")
}

const signedThinkingStream = "event: message_start\n" +
	`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[],"usage":{"input_tokens":3,"output_tokens":0}}}` + "\n\n" +
	"event: content_block_start\n" +
	`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}` + "\n\n" +
	"event: content_block_delta\n" +
	`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"check"}}` + "\n\n" +
	"event: content_block_delta\n" +
	`data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-1"}}` + "\n\n" +
	"event: content_block_stop\n" +
	`data: {"type":"content_block_stop","index":0}` + "\n\n" +
	"event: message_delta\n" +
	`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}` + "\n\n" +
	"event: message_stop\n" +
	`data: {"type":"message_stop"}` + "\n\n"

// Messages replies carry each thinking block's signature, so thinking from a
// backend that signs none reaches a Messages client with an empty one: on the
// buffered block, and in a stream on content_block_start and in the
// signature_delta that ends the block. A backend's own signature is the only
// one a stream sends, and text blocks carry none.
func TestThinkingReachesMessagesClientsWithASignature(t *testing.T) {
	translated, err := NewBuiltinEngine().TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, []byte(chatReasoningReply), nil)
	if err != nil {
		t.Fatal(err)
	}
	var message struct {
		Content []map[string]json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(translated.Body, &message); err != nil {
		t.Fatal(err)
	}
	if len(message.Content) != 2 || string(message.Content[0]["signature"]) != `""` || message.Content[1]["signature"] != nil {
		t.Fatalf("buffered reply = %s, want an empty signature on the thinking block only", translated.Body)
	}

	for _, test := range []struct {
		name    string
		source  llmprotocol.WireFormat
		payload []byte
		blocks  [][]string
	}{
		{"unsigned", llmprotocol.OpenAIChatV1, chatReasoningStream(), [][]string{
			{`start thinking signature=""`, "thinking_delta", `signature_delta ""`, "stop"},
			{"start text", "text_delta", "stop"},
		}},
		{"signed", llmprotocol.AnthropicMessagesV1, []byte(signedThinkingStream), [][]string{
			{`start thinking signature=""`, "thinking_delta", `signature_delta "sig-1"`, "stop"},
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			events := streamToMessagesClient(t, test.source, 0, test.payload)
			for index, want := range test.blocks {
				if got := blockEvents(events, index); !slices.Equal(got, want) {
					t.Errorf("block %d = %q, want %q", index, got, want)
				}
			}
		})
	}
}

// toolReplyFixture is the buffered reply toolStreamFixture streams.
func toolReplyFixture(format llmprotocol.WireFormat) []byte {
	switch format {
	case llmprotocol.OpenAIChatV1:
		return []byte(`{"id":"response_1","model":"source-model","choices":[{"index":0,"message":{"role":"assistant","content":null,` +
			`"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"protocol\":\"source\"}"}}]},` +
			`"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`)
	case llmprotocol.OpenAIResponsesV1:
		return []byte(`{"id":"response_1","model":"source-model","status":"completed","output":[{"type":"function_call","id":"item_1",` +
			`"call_id":"call_1","name":"lookup","arguments":"{\"protocol\":\"source\"}","status":"completed"}],` +
			`"usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3}}`)
	default:
		return []byte(`{"id":"response_1","type":"message","role":"assistant","model":"source-model",` +
			`"content":[{"type":"tool_use","id":"call_1","name":"lookup","input":{"protocol":"source"}}],` +
			`"stop_reason":"tool_use","usage":{"input_tokens":2,"output_tokens":1}}`)
	}
}

// A reply from a Chat or Responses backend reaches a Messages client with a
// msg_ message id and toolu_ tool-call ids derived from the backend's, the
// same buffered and streamed. A Messages backend's own ids are kept.
func TestMessagesClientsGetMessagesIdentifiers(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, source := range builtinFormats {
		t.Run(string(source), func(t *testing.T) {
			translated, err := engine.TranslateResponse(source, llmprotocol.AnthropicMessagesV1, toolReplyFixture(source), nil)
			if err != nil {
				t.Fatal(err)
			}
			var message struct {
				ID      string `json:"id"`
				Content []struct {
					ID string `json:"id"`
				} `json:"content"`
			}
			if err := json.Unmarshal(translated.Body, &message); err != nil || len(message.Content) != 1 {
				t.Fatalf("buffered reply = %s, %v", translated.Body, err)
			}
			messageID, toolID := message.ID, message.Content[0].ID
			if source == llmprotocol.AnthropicMessagesV1 {
				if messageID != "response_1" || toolID != "call_1" {
					t.Fatalf("a Messages backend's ids became %s and %s", messageID, toolID)
				}
			} else if !strings.HasPrefix(messageID, "msg_") || !strings.HasPrefix(toolID, "toolu_") {
				t.Fatalf("ids = %s and %s, want msg_ and toolu_ ones", messageID, toolID)
			}

			events := streamToMessagesClient(t, source, 0, toolStreamFixture(source))
			streamedToolID := ""
			for _, event := range events {
				if event.Type == "content_block_start" {
					streamedToolID = jsonText(event.ContentBlock["id"])
				}
			}
			if events[0].Message.ID != messageID || streamedToolID != toolID {
				t.Fatalf("streamed ids %s and %s, buffered %s and %s", events[0].Message.ID, streamedToolID, messageID, toolID)
			}
		})
	}
}

// A reply rendered again from the neutral response gets the ids translating
// it gives, and the caller's response keeps the backend's.
func TestAnthropicReplyIdentifiersMatchTranslation(t *testing.T) {
	engine := NewBuiltinEngine()
	ids := func(body []byte) string {
		var message struct {
			ID      string `json:"id"`
			Content []struct {
				ID string `json:"id"`
			} `json:"content"`
		}
		if err := json.Unmarshal(body, &message); err != nil || len(message.Content) != 1 {
			t.Fatalf("reply = %s, %v", body, err)
		}
		return message.ID + " " + message.Content[0].ID
	}
	for _, source := range builtinFormats {
		t.Run(string(source), func(t *testing.T) {
			translated, err := engine.TranslateResponse(source, llmprotocol.AnthropicMessagesV1, toolReplyFixture(source), nil)
			if err != nil {
				t.Fatal(err)
			}
			response, _, _, err := engine.DecodeResponse(source, toolReplyFixture(source))
			if err != nil {
				t.Fatal(err)
			}
			rendered, err := engine.EncodeResponse(llmprotocol.AnthropicMessagesV1, AnthropicReplyIdentifiers(source, response), llmprotocol.Envelope{})
			if err != nil {
				t.Fatal(err)
			}
			if got, want := ids(rendered.Body), ids(translated.Body); got != want {
				t.Fatalf("rendered ids %s, translated %s", got, want)
			}
			if response.ID != "response_1" || response.Output[0].Content[0].ToolCall.ID != "call_1" {
				t.Fatalf("the caller's response changed: %+v", response)
			}
		})
	}
}

func TestAnthropicReplyIDKeepsIdentifiersMessagesClientsAccept(t *testing.T) {
	chat := llmprotocol.OpenAIChatV1
	for _, test := range []struct {
		source llmprotocol.WireFormat
		id     string
	}{
		{chat, "toolu_01A09q90qw90lq917835lq9"},
		{chat, ""},
		{llmprotocol.AnthropicMessagesV1, "call_1"},
		{"", "call_1"},
	} {
		if got := anthropicReplyID(test.source, anthropicToolUseIDPrefix, test.id); got != test.id {
			t.Errorf("anthropicReplyID(%q, %q) = %q, want it kept", test.source, test.id, got)
		}
	}
	if anthropicReplyID(chat, anthropicToolUseIDPrefix, "call_1") == anthropicReplyID(chat, anthropicToolUseIDPrefix, "call_2") {
		t.Fatal("two calls got one id")
	}
}

// A Chat or Responses backend reports usage only when its stream ends, so a
// Messages client's message_start carries the Router's estimate of the input
// and the final message_delta the backend's count. Usage a backend reports by
// message_start wins over the estimate.
func TestMessageStartReportsTheEstimatedInput(t *testing.T) {
	for _, test := range []struct {
		source llmprotocol.WireFormat
		start  int64
	}{
		{llmprotocol.OpenAIChatV1, 42},
		{llmprotocol.OpenAIResponsesV1, 42},
		{llmprotocol.AnthropicMessagesV1, 2},
	} {
		t.Run(string(test.source), func(t *testing.T) {
			events := streamToMessagesClient(t, test.source, 42, toolStreamFixture(test.source))
			if events[0].Type != "message_start" || events[0].Message.Usage.InputTokens != test.start {
				t.Fatalf("first event %s reports %d input tokens, want message_start with %d", events[0].Type, events[0].Message.Usage.InputTokens, test.start)
			}
			final := events[len(events)-2]
			if final.Type != "message_delta" || final.Usage.InputTokens != 2 {
				t.Fatalf("%s reports %d input tokens, want message_delta with the backend's 2", final.Type, final.Usage.InputTokens)
			}
		})
	}
}

// A translated reply returns as history in the client's next turn. Its empty
// signature needs no signature support from the backend, and the tool result
// pairs with the call under the id the client was given.
func TestTranslatedReplyReturnsAsMessagesHistory(t *testing.T) {
	engine := NewBuiltinEngine()
	reply := `{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant",` +
		`"reasoning_content":"check","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup",` +
		`"arguments":"{\"city\":\"Paris\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`
	translated, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, []byte(reply), nil)
	if err != nil {
		t.Fatal(err)
	}
	var message struct {
		Content []json.RawMessage `json:"content"`
	}
	if err = json.Unmarshal(translated.Body, &message); err != nil || len(message.Content) != 2 {
		t.Fatalf("reply = %s, %v", translated.Body, err)
	}
	var call struct {
		ID string `json:"id"`
	}
	if err = json.Unmarshal(message.Content[1], &call); err != nil {
		t.Fatal(err)
	}
	content, err := json.Marshal(message.Content)
	if err != nil {
		t.Fatal(err)
	}
	history := fmt.Sprintf(`{"model":"m","max_tokens":64,"tools":[{"name":"lookup","input_schema":{"type":"object"}}],"messages":[`+
		`{"role":"user","content":"What is the weather in Paris?"},{"role":"assistant","content":%s},`+
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":%q,"content":"18C"}]}]}`, content, call.ID)
	request, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(history))
	if err != nil {
		t.Fatalf("history: %v", err)
	}
	if llmprotocol.RequiredCapabilities(request).Supports(llmprotocol.CapabilityReasoningSignature) {
		t.Fatal("an empty signature needs signature support")
	}
	sent, err := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, []byte(history), nil)
	if err != nil {
		t.Fatalf("history to a Chat backend: %v", err)
	}
	var chat struct {
		Messages []struct {
			ToolCallID string `json:"tool_call_id"`
			ToolCalls  []struct {
				ID string `json:"id"`
			} `json:"tool_calls"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(sent.Body, &chat); err != nil {
		t.Fatal(err)
	}
	if len(chat.Messages) != 3 || len(chat.Messages[1].ToolCalls) != 1 ||
		chat.Messages[1].ToolCalls[0].ID != call.ID || chat.Messages[2].ToolCallID != call.ID {
		t.Fatalf("Chat request = %s, want the call and its result under %s", sent.Body, call.ID)
	}
}
