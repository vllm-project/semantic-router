package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestAnthropicThinkingDisplaySurvivesSemanticMutation(t *testing.T) {
	tests := []struct {
		name string
		body string
		mode llmprotocol.ReasoningMode
	}{
		{
			name: "enabled",
			body: `{"model":"source-model","max_tokens":2048,"messages":[{"role":"user","content":"hello"}],"thinking":{"type":"enabled","budget_tokens":1024,"display":"summarized"}}`,
			mode: llmprotocol.ReasoningModeEnabled,
		},
		{
			name: "adaptive",
			body: `{"model":"source-model","max_tokens":64,"messages":[{"role":"user","content":"hello"}],"thinking":{"type":"adaptive","display":"omitted"}}`,
			mode: llmprotocol.ReasoningModeAdaptive,
		},
	}
	engine := NewBuiltinEngine()
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request, envelope, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(test.body))
			if err != nil {
				t.Fatal(err)
			}
			if request.ReasoningMode != test.mode || request.ReasoningDisplay == "" {
				t.Fatalf("decoded thinking = mode %q display %q", request.ReasoningMode, request.ReasoningDisplay)
			}
			request.Model = "routed-model"
			request.Generation++
			encoded, err := engine.EncodeRequest(llmprotocol.AnthropicMessagesV1, request, envelope)
			if err != nil {
				t.Fatal(err)
			}
			var wire anthropicRequestWire
			if err := json.Unmarshal(encoded.Body, &wire); err != nil {
				t.Fatal(err)
			}
			if wire.Thinking == nil || wire.Thinking.Type != string(test.mode) || wire.Thinking.Display != request.ReasoningDisplay {
				t.Fatalf("encoded thinking = %+v", wire.Thinking)
			}
		})
	}
}

func TestAnthropicThinkingValueDomainIsExplicit(t *testing.T) {
	engine := NewBuiltinEngine()
	invalid := []struct {
		name string
		body string
		code string
	}{
		{
			name: "budget below minimum",
			body: `{"model":"m","max_tokens":2048,"messages":[{"role":"user","content":"hello"}],"thinking":{"type":"enabled","budget_tokens":1023}}`,
			code: "invalid_anthropic_reasoning_budget",
		},
		{
			name: "budget equals max tokens",
			body: `{"model":"m","max_tokens":1024,"messages":[{"role":"user","content":"hello"}],"thinking":{"type":"enabled","budget_tokens":1024}}`,
			code: "invalid_anthropic_reasoning_budget",
		},
		{
			name: "enabled budget omitted",
			body: `{"model":"m","max_tokens":2048,"messages":[{"role":"user","content":"hello"}],"thinking":{"type":"enabled"}}`,
			code: "reasoning_budget_required",
		},
		{
			name: "adaptive carries budget",
			body: `{"model":"m","max_tokens":2048,"messages":[{"role":"user","content":"hello"}],"thinking":{"type":"adaptive","budget_tokens":1024}}`,
			code: "invalid_adaptive_thinking",
		},
		{
			name: "disabled carries display",
			body: `{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hello"}],"thinking":{"type":"disabled","display":"omitted"}}`,
			code: "invalid_disabled_thinking",
		},
		{
			name: "display enum invalid",
			body: `{"model":"m","max_tokens":2048,"messages":[{"role":"user","content":"hello"}],"thinking":{"type":"enabled","budget_tokens":1024,"display":"verbose"}}`,
			code: "invalid_reasoning_display",
		},
		{
			name: "type omitted",
			body: `{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hello"}],"thinking":{}}`,
			code: "thinking_type_required",
		},
	}
	for _, test := range invalid {
		t.Run(test.name, func(t *testing.T) {
			_, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(test.body))
			assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, test.code)
		})
	}

	valid := []string{
		`{"model":"m","max_tokens":2048,"messages":[{"role":"user","content":"hello"}],"thinking":{"type":"enabled","budget_tokens":1024,"display":"summarized"}}`,
		`{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hello"}],"thinking":{"type":"adaptive","display":"omitted"}}`,
	}
	for index, body := range valid {
		if _, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(body)); err != nil {
			t.Fatalf("valid case %d rejected: %v", index, err)
		}
	}
}

func TestAnthropicZeroMaxTokensIsProviderSpecific(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"model":"source-model","max_tokens":0,"messages":[{"role":"user","content":"warm the prompt cache"}]}`)
	request, envelope, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	if request.Sampling.MaxOutputTokens == nil || *request.Sampling.MaxOutputTokens != 0 {
		t.Fatalf("max output tokens = %v, want explicit zero", request.Sampling.MaxOutputTokens)
	}
	request.Model = "routed-model"
	request.Generation++
	encoded, err := engine.EncodeRequest(llmprotocol.AnthropicMessagesV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	var wire anthropicRequestWire
	if err := json.Unmarshal(encoded.Body, &wire); err != nil {
		t.Fatal(err)
	}
	if wire.MaxTokens == nil || *wire.MaxTokens != 0 {
		t.Fatalf("max_tokens = %v, want explicit 0", wire.MaxTokens)
	}
	chat, err := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, body, func(request *llmprotocol.Request) error {
		request.Model = "routed-model"
		return nil
	})
	if err != nil {
		t.Fatalf("Chat Completions should preserve a zero token limit: %v", err)
	}
	var chatWire chatRequestWire
	if err := json.Unmarshal(chat.Body, &chatWire); err != nil {
		t.Fatal(err)
	}
	if chatWire.MaxCompletionTokens == nil || *chatWire.MaxCompletionTokens != 0 {
		t.Fatalf("Chat max_completion_tokens = %v, want explicit zero", chatWire.MaxCompletionTokens)
	}
	_, err = engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1, body, func(request *llmprotocol.Request) error {
		request.Model = "routed-model"
		return nil
	})
	assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_responses_max_output_tokens")
}

func TestOpenAIOutputTokenLimitValueDomains(t *testing.T) {
	engine := NewBuiltinEngine()
	tests := []struct {
		name   string
		format llmprotocol.WireFormat
		body   string
		code   string
	}{
		{
			name:   "chat max tokens",
			format: llmprotocol.OpenAIChatV1,
			body:   `{"model":"m","max_tokens":-1,"messages":[{"role":"user","content":"hello"}]}`,
			code:   "invalid_chat_max_output_tokens",
		},
		{
			name:   "chat max completion tokens",
			format: llmprotocol.OpenAIChatV1,
			body:   `{"model":"m","max_completion_tokens":-1,"messages":[{"role":"user","content":"hello"}]}`,
			code:   "invalid_chat_max_output_tokens",
		},
		{
			name:   "responses zero max output tokens",
			format: llmprotocol.OpenAIResponsesV1,
			body:   `{"model":"m","max_output_tokens":0,"input":"hello"}`,
			code:   "invalid_responses_max_output_tokens",
		},
		{
			name:   "responses below minimum max output tokens",
			format: llmprotocol.OpenAIResponsesV1,
			body:   `{"model":"m","max_output_tokens":15,"input":"hello"}`,
			code:   "invalid_responses_max_output_tokens",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, _, _, err := engine.DecodeRequest(test.format, []byte(test.body))
			assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, test.code)
		})
	}
}

func TestSmallOpenAIReasoningBudgetCannotTargetAnthropic(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"hello"}],"reasoning_budget_tokens":512}`)
	_, err := engine.TranslateRequest(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, body, nil)
	assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_anthropic_reasoning_budget")
}
