package protocolcodec

import (
	"encoding/json"
	"errors"
	"os"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The two /v1/responses requests Codex CLI 0.156.1 sends for a task with one
// tool call, captured with multi-agent tools, web search, and reasoning
// summaries turned off in Codex. Prompt text is shortened; IDs are placeholders.
const codexToolLoopFixture = "testdata/clients/codex-cli-0.156.1-tool-loop.json"

const codexPromptCacheKey = "00000000-0000-4000-8000-000000000004"

func loadCodexToolLoop(t *testing.T) []json.RawMessage {
	t.Helper()
	raw, err := os.ReadFile(codexToolLoopFixture)
	if err != nil {
		t.Fatal(err)
	}
	var turns []json.RawMessage
	if err := json.Unmarshal(raw, &turns); err != nil {
		t.Fatal(err)
	}
	if len(turns) != 2 {
		t.Fatalf("fixture has %d turns, want 2", len(turns))
	}
	return turns
}

func TestCodexToolLoopDecodesUnderEitherLossyPolicy(t *testing.T) {
	for _, lossy := range []llmprotocol.LossyPolicy{llmprotocol.LossyReject, llmprotocol.LossyAllowWithDiagnostic} {
		policy := llmprotocol.DefaultPolicy()
		policy.LossyFeatures = lossy
		engine, err := NewEngine(NewBuiltinRegistry(), policy)
		if err != nil {
			t.Fatal(err)
		}
		for turn, body := range loadCodexToolLoop(t) {
			_, _, diagnostics, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, body)
			if err != nil {
				t.Fatalf("lossy=%s turn %d: Codex request rejected: %v", lossy, turn+1, err)
			}
			assertDroppedFields(t, diagnostics, "client_metadata", "include")
		}
	}
}

func TestCodexToolLoopDispatchDropsPromptCacheKeyOnlyForMessages(t *testing.T) {
	engine := NewBuiltinEngine()
	for turn, body := range loadCodexToolLoop(t) {
		request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, body)
		if err != nil {
			t.Fatalf("turn %d: %v", turn+1, err)
		}
		// Routing rewrites the model, and the Router clears its own storage
		// controls before dispatch, as materializeResponseObjectContext does.
		request.Model = "routed-model"
		request.Store = nil
		request.Generation++
		for _, format := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
			encoded, encodeErr := engine.EncodeRequest(format, request, envelope)
			if encodeErr != nil {
				t.Fatalf("turn %d to %s: %v", turn+1, format, encodeErr)
			}
			var dispatch map[string]json.RawMessage
			if unmarshalErr := json.Unmarshal(encoded.Body, &dispatch); unmarshalErr != nil {
				t.Fatal(unmarshalErr)
			}
			if string(dispatch["prompt_cache_key"]) != `"`+codexPromptCacheKey+`"` {
				t.Fatalf("turn %d to %s: prompt_cache_key = %s", turn+1, format, dispatch["prompt_cache_key"])
			}
			for _, dropped := range []string{"client_metadata", "include"} {
				if _, ok := dispatch[dropped]; ok {
					t.Fatalf("turn %d to %s forwarded %s: %s", turn+1, format, dropped, encoded.Body)
				}
			}
		}
		encoded, err := engine.EncodeRequest(llmprotocol.AnthropicMessagesV1, request, envelope)
		if err != nil {
			t.Fatalf("turn %d to Messages: %v", turn+1, err)
		}
		var dispatch map[string]json.RawMessage
		if err := json.Unmarshal(encoded.Body, &dispatch); err != nil {
			t.Fatal(err)
		}
		if _, forwarded := dispatch["prompt_cache_key"]; forwarded {
			t.Fatalf("turn %d leaked prompt_cache_key to Messages: %s", turn+1, encoded.Body)
		}
		assertDroppedFields(t, encoded.Diagnostics, "prompt_cache_key")
	}
}

func TestCodexToolResultTurnReachesChatAsToolMessages(t *testing.T) {
	engine := NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, loadCodexToolLoop(t)[1])
	if err != nil {
		t.Fatal(err)
	}
	request.Model = "routed-model"
	request.Generation++
	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	var chat chatRequestWire
	if err := json.Unmarshal(encoded.Body, &chat); err != nil {
		t.Fatal(err)
	}
	last := len(chat.Messages) - 1
	call, result := chat.Messages[last-1], chat.Messages[last]
	if call.Role != "assistant" || len(call.ToolCalls) != 1 || call.ToolCalls[0].ID != "call_1" ||
		call.ToolCalls[0].Function.Name != "exec_command" {
		t.Fatalf("tool call did not reach Chat: %+v", call)
	}
	if result.Role != "tool" || result.ToolCallID != "call_1" {
		t.Fatalf("tool result did not reach Chat: %+v", result)
	}
}

func TestResponsesIncludeAcceptsOnlyEncryptedReasoning(t *testing.T) {
	tests := []struct {
		include  string
		category llmprotocol.ErrorCategory
		dropped  bool
	}{
		{include: `[]`},
		{include: `null`},
		{include: `["reasoning.encrypted_content"]`, dropped: true},
		{include: `["message.output_text.logprobs"]`, category: llmprotocol.ErrorUnsupportedFeature},
		{include: `["reasoning.encrypted_content","web_search_call.results"]`, category: llmprotocol.ErrorUnsupportedFeature},
		{include: `"reasoning.encrypted_content"`, category: llmprotocol.ErrorInvalidRequest},
	}
	engine := NewBuiltinEngine()
	for _, test := range tests {
		t.Run(test.include, func(t *testing.T) {
			body := []byte(`{"model":"m","input":"hello","include":` + test.include + `}`)
			_, _, diagnostics, err := engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, body)
			if test.category != "" {
				var protocolError *llmprotocol.ProtocolError
				if !errors.As(err, &protocolError) || protocolError.Category != test.category {
					t.Fatalf("include %s returned %v, want %s", test.include, err, test.category)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if test.dropped {
				assertDroppedFields(t, diagnostics, "include")
			} else if len(diagnostics) != 0 {
				t.Fatalf("include %s produced diagnostics %+v", test.include, diagnostics)
			}
		})
	}
}

func assertDroppedFields(t *testing.T, diagnostics llmprotocol.Diagnostics, fields ...string) {
	t.Helper()
	dropped := make(map[string]bool, len(diagnostics))
	for _, diagnostic := range diagnostics {
		if diagnostic.Action == llmprotocol.DiagnosticDropped {
			dropped[diagnostic.Field] = true
		}
	}
	for _, field := range fields {
		if !dropped[field] {
			t.Fatalf("%s was not reported as dropped: %+v", field, diagnostics)
		}
	}
}
