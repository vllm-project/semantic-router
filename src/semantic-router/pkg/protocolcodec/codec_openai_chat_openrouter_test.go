package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func openRouterFixture(t *testing.T, name string) []byte {
	t.Helper()
	body, err := os.ReadFile(filepath.Join("testdata", "contracts", name))
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func TestOpenRouterBufferedReplyTranslatesAcrossClientFormats(t *testing.T) {
	body := openRouterFixture(t, "openrouter-chat-response-in.json")
	for _, target := range []llmprotocol.WireFormat{
		llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1,
	} {
		t.Run(string(target), func(t *testing.T) {
			result, err := NewBuiltinEngine().TranslateResponse(llmprotocol.OpenAIChatV1, target, body, nil)
			if err != nil {
				t.Fatalf("TranslateResponse(%s): %v", target, err)
			}
			if len(result.Response.Output) != 1 || result.Response.Usage.Total.Value == nil ||
				*result.Response.Usage.Total.Value != 13 {
				t.Fatalf("translated response = %+v", result.Response)
			}
			forbiddenFields := []string{`"provider":`, `"native_finish_reason":`, `"cost":`, `"is_byok":`, `"cost_details":`, `"video_tokens":`, `"image_tokens":`}
			if target != llmprotocol.AnthropicMessagesV1 {
				forbiddenFields = append(forbiddenFields, `"server_tool_use":`)
			}
			for _, forbidden := range forbiddenFields {
				if bytes.Contains(result.Body, []byte(forbidden)) {
					t.Errorf("public %s reply leaked %s: %s", target, forbidden, result.Body)
				}
			}
			expectedDiagnostics := []string{
				"provider", "choices.native_finish_reason", "usage.cost", "usage.is_byok",
				"usage.cost_details", "usage.server_tool_use", "usage.prompt_tokens_details.video_tokens",
				"usage.completion_tokens_details.image_tokens",
			}
			if target == llmprotocol.AnthropicMessagesV1 {
				expectedDiagnostics = append(expectedDiagnostics, "usage.cache")
			}
			assertDiagnosticFields(t, result.Diagnostics, expectedDiagnostics...)
		})
	}
}

func TestOpenRouterBufferedReplyKeepsUnknownFieldsClosed(t *testing.T) {
	body := string(openRouterFixture(t, "openrouter-chat-response-in.json"))
	for name, changed := range map[string]string{
		"top_level":    strings.Replace(body, `"provider": "OpenAI",`, `"provider": "OpenAI", "unknown": true,`, 1),
		"cost_details": strings.Replace(body, `"server_tool_cost": null`, `"server_tool_cost": null, "unknown": true`, 1),
	} {
		t.Run(name, func(t *testing.T) {
			_, _, _, err := NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIChatV1, []byte(changed))
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json")
		})
	}
}

func TestOpenRouterStreamRepeatFinishTranslatesAcrossClientFormats(t *testing.T) {
	var fixture struct {
		Chunks []json.RawMessage `json:"chunks"`
	}
	if err := json.Unmarshal(openRouterFixture(t, "openrouter-chat-stream-in.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Chunks) != 3 {
		t.Fatalf("stream fixture chunks = %d, want 3", len(fixture.Chunks))
	}
	for _, target := range []llmprotocol.WireFormat{
		llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1,
	} {
		t.Run(string(target), func(t *testing.T) {
			stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIChatV1, target,
				llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model", ProviderModel: "openai/gpt-4.1-nano"})
			if err != nil {
				t.Fatal(err)
			}
			var public []byte
			var diagnostics llmprotocol.Diagnostics
			completionCount := 0
			itemCompletionCount := 0
			for _, chunk := range fixture.Chunks {
				var compact bytes.Buffer
				if compactErr := json.Compact(&compact, chunk); compactErr != nil {
					t.Fatal(compactErr)
				}
				frames, events, observed, pushErr := stream.Push(append(append([]byte("data: "), compact.Bytes()...), []byte("\n\n")...))
				if pushErr != nil {
					t.Fatalf("Push() error = %v", pushErr)
				}
				public = append(public, bytes.Join(frames, nil)...)
				diagnostics = append(diagnostics, observed...)
				for _, event := range events {
					if event.Type == llmprotocol.EventOutputItemCompleted {
						itemCompletionCount++
					}
					if event.Type == llmprotocol.EventResponseCompleted {
						completionCount++
					}
				}
			}
			frames, events, observed, err := stream.Push([]byte("data: [DONE]\n\n"))
			if err != nil {
				t.Fatalf("Push([DONE]) error = %v", err)
			}
			public = append(public, bytes.Join(frames, nil)...)
			diagnostics = append(diagnostics, observed...)
			for _, event := range events {
				if event.Type == llmprotocol.EventResponseCompleted {
					completionCount++
				}
			}
			frames, events, observed, err = stream.Finalize(nil)
			if err != nil {
				t.Fatalf("Finalize() error = %v", err)
			}
			public = append(public, bytes.Join(frames, nil)...)
			diagnostics = append(diagnostics, observed...)
			for _, event := range events {
				if event.Type == llmprotocol.EventResponseCompleted {
					completionCount++
				}
			}
			if completionCount != 1 || itemCompletionCount != 1 {
				t.Fatalf("terminal event counts: response=%d item=%d", completionCount, itemCompletionCount)
			}
			for _, forbidden := range []string{`"provider":`, `"native_finish_reason":`, `"cost":`, `"is_byok":`, `"cost_details":`} {
				if bytes.Contains(public, []byte(forbidden)) {
					t.Errorf("public %s stream leaked %s: %s", target, forbidden, public)
				}
			}
			expectedDiagnostics := []string{
				"stream.provider", "stream.choices.native_finish_reason", "stream.usage.cost",
				"stream.usage.is_byok", "stream.usage.cost_details", "stream.usage.server_tool_use",
				"stream.usage.prompt_tokens_details.video_tokens", "stream.usage.completion_tokens_details.image_tokens",
			}
			if target == llmprotocol.AnthropicMessagesV1 {
				expectedDiagnostics = append(expectedDiagnostics, "usage.cache")
			}
			assertDiagnosticFields(t, diagnostics, expectedDiagnostics...)
		})
	}
}
