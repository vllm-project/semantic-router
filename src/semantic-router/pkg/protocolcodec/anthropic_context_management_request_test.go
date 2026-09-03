package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Claude Code sends a top-level context_management field on every /v1/messages
// request. It asks the upstream Anthropic API to trim stale thinking from the
// billed prompt, so it changes what the provider charges for the turn and must
// survive to an Anthropic-format target. No OpenAI wire can carry it, so
// cross-format translation omits it with an explicit diagnostic.

// contextManagement is the field under test: the server-side prompt-trimming
// directive that must round-trip to an Anthropic target unchanged.
const contextManagement = `"context_management":{"edits":[{"type":"clear_thinking_20251015","keep":"all"}]}`

func anthropicRequestBodyWithContextManagement() []byte {
	return []byte(`{"model":"source-model","max_tokens":64,` +
		`"messages":[{"role":"user","content":"hello"}],` + contextManagement + `}`)
}

func TestAnthropicRequestContextManagementRoundTripsToAnthropicTarget(t *testing.T) {
	engine := NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequest(
		llmprotocol.AnthropicMessagesV1, anthropicRequestBodyWithContextManagement(),
	)
	if err != nil {
		t.Fatalf("request carrying context_management was rejected: %v", err)
	}

	// Bump the generation so the encode rebuilds the wire from the neutral
	// request instead of replaying the captured bytes; byte replay would
	// preserve the field trivially and prove nothing about the model carry.
	request.Generation++
	roundTrip, err := engine.EncodeRequest(llmprotocol.AnthropicMessagesV1, request, envelope)
	if err != nil {
		t.Fatalf("re-encoding for an Anthropic target failed: %v", err)
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(roundTrip.Body, &object); err != nil {
		t.Fatalf("re-encoded Anthropic body is not a JSON object: %v\n%s", err, roundTrip.Body)
	}
	want := json.RawMessage(`{"edits":[{"type":"clear_thinking_20251015","keep":"all"}]}`)
	if !jsonSemanticallyEqual(object["context_management"], want) {
		t.Fatalf("context_management did not round-trip to the Anthropic target: %s", roundTrip.Body)
	}
}

func TestAnthropicRequestContextManagementOmissionIsDiagnosedForOpenAITargets(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		t.Run(string(target), func(t *testing.T) {
			translated, err := engine.TranslateRequest(
				llmprotocol.AnthropicMessagesV1,
				target,
				anthropicRequestBodyWithContextManagement(),
				func(request *llmprotocol.Request) error { request.Model = "routed-model"; return nil },
			)
			if err != nil {
				t.Fatalf("translating a request carrying context_management to %s failed: %v", target, err)
			}
			var object map[string]json.RawMessage
			if err := json.Unmarshal(translated.Body, &object); err != nil {
				t.Fatalf("translated %s body is not a JSON object: %v\n%s", target, err, translated.Body)
			}
			if _, present := object["context_management"]; present {
				t.Fatalf("context_management leaked into the %s target: %s", target, translated.Body)
			}
			assertDiagnosticFields(t, translated.Diagnostics, "context_management")
		})
	}
}
