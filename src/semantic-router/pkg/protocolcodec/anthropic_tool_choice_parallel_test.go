package protocolcodec

import (
	"bytes"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestAnthropicToolChoiceParallelControl(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, source := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		for _, mode := range []llmprotocol.ToolChoiceMode{"", llmprotocol.ToolChoiceNone, llmprotocol.ToolChoiceAuto, llmprotocol.ToolChoiceRequired, llmprotocol.ToolChoiceNamed} {
			for _, parallel := range []string{"omitted", "false", "true"} {
				t.Run(string(source)+"/"+string(mode)+"/"+parallel, func(t *testing.T) {
					var body map[string]any
					if err := json.Unmarshal(toolChoiceFixture(source, mode), &body); err != nil {
						t.Fatal(err)
					}
					if mode == "" {
						delete(body, "tool_choice")
					}
					if parallel != "omitted" {
						body["parallel_tool_calls"] = parallel == "true"
					}
					input, err := json.Marshal(body)
					if err != nil {
						t.Fatal(err)
					}
					original := bytes.Clone(input)
					translated, err := engine.TranslateRequest(source, llmprotocol.AnthropicMessagesV1, input, nil)
					if err != nil {
						t.Fatal(err)
					}
					if !bytes.Equal(input, original) {
						t.Fatal("translation mutated source bytes")
					}
					if parallel != "omitted" && (translated.Request.ParallelToolCalls == nil || *translated.Request.ParallelToolCalls != (parallel == "true")) {
						t.Fatal("encoding changed neutral parallel control")
					}
					var wire struct {
						ToolChoice map[string]any `json:"tool_choice"`
					}
					if err := json.Unmarshal(translated.Body, &wire); err != nil {
						t.Fatal(err)
					}
					wantType := map[llmprotocol.ToolChoiceMode]string{"": "auto", llmprotocol.ToolChoiceAuto: "auto", llmprotocol.ToolChoiceNone: "none", llmprotocol.ToolChoiceRequired: "any", llmprotocol.ToolChoiceNamed: "tool"}[mode]
					want := map[string]any{"type": wantType}
					if mode == llmprotocol.ToolChoiceNamed {
						want["name"] = "lookup"
					}
					if mode != llmprotocol.ToolChoiceNone && parallel != "omitted" {
						want["disable_parallel_tool_use"] = parallel == "false"
					}
					if !reflect.DeepEqual(wire.ToolChoice, want) {
						t.Fatalf("tool_choice = %v, want %v", wire.ToolChoice, want)
					}
					if _, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, translated.Body); err != nil {
						t.Fatalf("generated Anthropic request is invalid: %v", err)
					}
				})
			}
		}
	}
}

func TestAnthropicNoneStillRejectsParallelControlOnIngress(t *testing.T) {
	for _, value := range []string{"false", "true"} {
		body := []byte(`{"model":"m","max_tokens":32,"messages":[{"role":"user","content":"hello"}],"tool_choice":{"type":"none","disable_parallel_tool_use":` + value + `}}`)
		if _, _, _, err := NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1, body); err == nil {
			t.Fatalf("accepted invalid native Anthropic none variant with %s", value)
		}
	}
}
