package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Claude Code sends a top-level context_management field on every /v1/messages
// request. It asks the upstream Anthropic API to trim stale thinking from the
// billed prompt, so it changes what the provider charges for the turn and must
// survive to an Anthropic-format target. No OpenAI wire represents it, so a
// cross-format target must omit it. The neutral contract therefore carries the
// field opaquely rather than rejecting it under strict unknown-field decode.

// contextManagement is the field under test: the server-side prompt-trimming
// directive that must round-trip to an Anthropic target unchanged.
const contextManagement = `"context_management":{"edits":[{"type":"clear_thinking_20251015","keep":"all"}]}`

func TestAnthropicRequestContextManagementRoundTripsToAnthropicTarget(t *testing.T) {
	engine := NewBuiltinEngine()
	// A minimal Anthropic Messages request with the directive embedded in the body.
	body := []byte(`{"model":"source-model","max_tokens":64,` +
		`"messages":[{"role":"user","content":"hello"}],` + contextManagement + `}`)

	request, envelope, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, body)
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

func TestAnthropicRequestContextManagementIsOmittedForChatTarget(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"model":"source-model","max_tokens":64,` +
		`"messages":[{"role":"user","content":"hello"}],` + contextManagement + `}`)

	translated, err := engine.TranslateRequest(
		llmprotocol.AnthropicMessagesV1,
		llmprotocol.OpenAIChatV1,
		body,
		func(request *llmprotocol.Request) error { request.Model = "routed-model"; return nil },
	)
	if err != nil {
		t.Fatalf("translating a request carrying context_management to Chat failed: %v", err)
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(translated.Body, &object); err != nil {
		t.Fatalf("translated Chat body is not a JSON object: %v\n%s", err, translated.Body)
	}
	if _, present := object["context_management"]; present {
		t.Fatalf("context_management leaked into the Chat target: %s", translated.Body)
	}
}
