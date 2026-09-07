package protocolcodec

import (
	"bytes"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func hasDiagnostic(
	diagnostics llmprotocol.Diagnostics,
	field string,
	action llmprotocol.DiagnosticAction,
) bool {
	for _, diagnostic := range diagnostics {
		if diagnostic.Field == field && diagnostic.Action == action {
			return true
		}
	}
	return false
}

// The Responses API writes its system prompt as a developer instruction with no
// alternative, so rejecting that role made every such request fail on Anthropic.
func TestDeveloperInstructionsReachAnthropic(t *testing.T) {
	engine := NewBuiltinEngine()
	cases := []struct {
		name   string
		source llmprotocol.WireFormat
		body   string
	}{
		{
			name:   "chat developer role",
			source: llmprotocol.OpenAIChatV1,
			body: `{"model":"m","messages":[` +
				`{"role":"developer","content":"be terse"},` +
				`{"role":"user","content":"hi"}]}`,
		},
		{
			name:   "responses instructions",
			source: llmprotocol.OpenAIResponsesV1,
			body: `{"model":"m","instructions":"be terse",` +
				`"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`,
		},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			result, err := engine.TranslateRequest(
				testCase.source, llmprotocol.AnthropicMessagesV1, []byte(testCase.body), nil)
			if err != nil {
				t.Fatalf("translate: %v", err)
			}
			if !bytes.Contains(result.Body, []byte(`"system"`)) {
				t.Fatalf("instructions did not reach the system block: %s", result.Body)
			}
			if !bytes.Contains(result.Body, []byte("be terse")) {
				t.Fatalf("instruction text was dropped: %s", result.Body)
			}
			if !hasDiagnostic(result.Diagnostics, "instructions.role", llmprotocol.DiagnosticApproximated) {
				t.Fatalf("no approximation diagnostic recorded: %+v", result.Diagnostics)
			}
		})
	}
}

// A system-role request must be unaffected, and must not gain the diagnostic.
func TestSystemInstructionsUnchangedOnAnthropic(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"model":"m","messages":[` +
		`{"role":"system","content":"be terse"},` +
		`{"role":"user","content":"hi"}]}`)

	result, err := engine.TranslateRequest(
		llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, body, nil)
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	if !bytes.Contains(result.Body, []byte("be terse")) {
		t.Fatalf("instruction text was dropped: %s", result.Body)
	}
	if hasDiagnostic(result.Diagnostics, "instructions.role", llmprotocol.DiagnosticApproximated) {
		t.Fatalf("system role should not report an approximation: %+v", result.Diagnostics)
	}
}

// Developer role stays a distinct role when the target can express it.
func TestDeveloperRolePreservedOnOpenAITargets(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"model":"m","messages":[` +
		`{"role":"developer","content":"be terse"},` +
		`{"role":"user","content":"hi"}]}`)

	result, err := engine.TranslateRequest(
		llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, body,
		func(request *llmprotocol.Request) error { request.Model = "other"; return nil })
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	if !bytes.Contains(result.Body, []byte(`"role":"developer"`)) {
		t.Fatalf("developer role was not preserved: %s", result.Body)
	}
}

// Genuinely unrepresentable features must still fail under LossyReject.
func TestUnrepresentableFeaturesStillRejected(t *testing.T) {
	engine := NewBuiltinEngine()
	refusal := []byte(`{"id":"r1","model":"m","choices":[{"index":0,` +
		`"message":{"role":"assistant","refusal":"no"},"finish_reason":"content_filter"}],` +
		`"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`)

	if _, err := engine.TranslateResponse(
		llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, refusal, nil); err == nil {
		t.Fatal("refusal semantics were silently converted to text")
	}
}
