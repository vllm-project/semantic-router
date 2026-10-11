package protocolcodec

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestChatToAnthropicEmptyRequestText(t *testing.T) {
	const call = `"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{\"x\":1}"}}]`
	const use = `{"type":"tool_use","id":"c1","name":"f","input":{"x":1}}`
	const user = `{"role":"user","content":[{"type":"text","text":"hi"}]}`
	cases := []struct{ name, messages, want string }{
		{
			name:     "assistant empty string",
			messages: `[{"role":"user","content":"hi"},{"role":"assistant","content":"",` + call + `}]`,
			want:     `{"messages":[` + user + `,{"role":"assistant","content":[` + use + `]}]}`,
		},
		{
			name:     "assistant null",
			messages: `[{"role":"user","content":"hi"},{"role":"assistant","content":null,` + call + `}]`,
			want:     `{"messages":[` + user + `,{"role":"assistant","content":[` + use + `]}]}`,
		},
		{
			name:     "mixed assistant blocks",
			messages: `[{"role":"user","content":"hi"},{"role":"assistant","content":[{"type":"text","text":""},{"type":"text","text":"checking"},{"type":"text","text":""}],` + call + `}]`,
			want:     `{"messages":[` + user + `,{"role":"assistant","content":[{"type":"text","text":"checking"},` + use + `]}]}`,
		},
		{
			name:     "empty tool result",
			messages: `[{"role":"user","content":"hi"},{"role":"assistant","content":"",` + call + `},{"role":"tool","tool_call_id":"c1","content":""}]`,
			want:     `{"messages":[` + user + `,{"role":"assistant","content":[` + use + `]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"c1","content":[]}]}]}`,
		},
		{
			name:     "nonempty tool result",
			messages: `[{"role":"user","content":"hi"},{"role":"assistant","content":"",` + call + `},{"role":"tool","tool_call_id":"c1","content":"ok"}]`,
			want:     `{"messages":[` + user + `,{"role":"assistant","content":[` + use + `]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"c1","content":[{"type":"text","text":"ok"}]}]}]}`,
		},
		{
			name:     "empty system",
			messages: `[{"role":"system","content":""},{"role":"user","content":"hi"}]`,
			want:     `{"messages":[` + user + `]}`,
		},
		{
			name:     "mixed instructions",
			messages: `[{"role":"system","content":""},{"role":"developer","content":"be terse"},{"role":"user","content":"hi"}]`,
			want:     `{"system":[{"type":"text","text":"be terse"}],"messages":[` + user + `]}`,
		},
		{
			name:     "whitespace preserved",
			messages: `[{"role":"user","content":"hi"},{"role":"assistant","content":" ",` + call + `}]`,
			want:     `{"messages":[` + user + `,{"role":"assistant","content":[{"type":"text","text":" "},` + use + `]}]}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := NewBuiltinEngine().TranslateRequest(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, []byte(`{"model":"m","max_tokens":20,"messages":`+tc.messages+`}`), nil)
			if err != nil {
				t.Fatal(err)
			}
			var got, want map[string]any
			if err = json.Unmarshal(result.Body, &got); err != nil {
				t.Fatal(err)
			}
			if err = json.Unmarshal([]byte(tc.want), &want); err != nil {
				t.Fatal(err)
			}
			want["model"], want["max_tokens"] = "m", float64(20)
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("got %s; want %+v", result.Body, want)
			}
			if _, _, _, err := (AnthropicMessagesCodec{}).DecodeRequest(result.Body, llmprotocol.DefaultPolicy()); err != nil {
				t.Fatalf("translated request cannot be decoded: %v", err)
			}
		})
	}
}

func TestAnthropicEmptyTextDoesNotEraseMessageOrCacheBoundary(t *testing.T) {
	for _, role := range []string{"user", "assistant"} {
		t.Run(role, func(t *testing.T) {
			body := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"},{"role":"` + role + `","content":""}]}`)
			_, err := NewBuiltinEngine().TranslateRequest(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, body, nil)
			assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "empty_anthropic_message")
		})
	}
	body := []byte(`{"model":"m","max_tokens":20,"system":[{"type":"text","text":"","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":"hi"}]}`)
	_, err := NewBuiltinEngine().TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, body,
		func(request *llmprotocol.Request) error { request.Model = "other"; return nil })
	assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_cache_directive")
}

func TestAnthropicEmptyTextNormalizationIsRequestOnly(t *testing.T) {
	body := []byte(`{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":0,"total_tokens":1}}`)
	result, err := NewBuiltinEngine().TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, body, nil)
	if err != nil {
		t.Fatal(err)
	}
	var wire struct{ Content []struct{ Type, Text string } }
	if err = json.Unmarshal(result.Body, &wire); err != nil {
		t.Fatal(err)
	}
	if len(wire.Content) != 1 || wire.Content[0].Type != "text" || wire.Content[0].Text != "" {
		t.Fatalf("empty response changed: %s", result.Body)
	}
}

func TestAnthropicEmptyHistoryTurnIsNotSilentlyDropped(t *testing.T) {
	for _, messages := range []string{
		`[{"role":"user","content":"a"},{"role":"assistant","content":""},{"role":"user","content":"b"}]`,
		`[{"role":"user","content":""}]`,
	} {
		_, err := NewBuiltinEngine().TranslateRequest(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1,
			[]byte(`{"model":"m","max_tokens":20,"messages":`+messages+`}`), nil)
		assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "empty_anthropic_message")
	}
}
