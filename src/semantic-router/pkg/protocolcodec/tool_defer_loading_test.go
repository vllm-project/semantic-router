package protocolcodec

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Claude Code marks some tools for tool search. Strict translation keeps refusing
// the flag. With tools.defer_loading accepted, a request the Router re-encodes
// (here a routed model name) reaches the backend with every tool's full
// definition and no flag, and the loss is reported once per request.
func TestDeferLoadingOptIn(t *testing.T) {
	anthropic := `{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hi"}],"tools":[
		{"name":"Bash","input_schema":{"type":"object"}},
		{"name":"CronCreate","input_schema":{"type":"object"},"defer_loading":true},
		{"name":"WebFetch","input_schema":{"type":"object"},"defer_loading":true}]}`
	responses := `{"model":"m","input":"hi","tools":[
		{"type":"function","name":"lookup","parameters":{"type":"object"},"defer_loading":true},
		{"type":"custom","name":"patch","defer_loading":true}]}`
	namespaced := `{"model":"m","input":"hi","tools":[
		{"type":"namespace","name":"mcp__echo","description":"Echo server","tools":[
			{"type":"function","name":"echo","parameters":{"type":"object"},"defer_loading":true}]}]}`
	cases := []struct {
		source, target llmprotocol.WireFormat
		body           string
		tools          int
	}{
		{llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, anthropic, 3},
		{llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, anthropic, 3},
		{llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1, responses, 2},
		{llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, namespaced, 1},
	}
	for _, tc := range cases {
		t.Run(string(tc.source)+"->"+string(tc.target), func(t *testing.T) {
			strict, err := NewEngine(NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
			if err != nil {
				t.Fatal(err)
			}
			route := func(request *llmprotocol.Request) error { request.Model = "routed"; return nil }
			_, err = strict.TranslateRequest(tc.source, tc.target, []byte(tc.body), route)
			assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_tools_defer_loading")

			policy := llmprotocol.DefaultPolicy()
			policy.AllowedLossyFields = []string{llmprotocol.LossyFieldToolDeferLoading}
			engine, err := NewEngine(NewBuiltinRegistry(), policy)
			if err != nil {
				t.Fatal(err)
			}
			result, err := engine.TranslateRequest(tc.source, tc.target, []byte(tc.body), route)
			if err != nil {
				t.Fatalf("opted-in translation failed: %v", err)
			}
			if bytes.Contains(result.Body, []byte("defer_loading")) {
				t.Fatalf("backend body still carries defer_loading: %s", result.Body)
			}
			var body struct {
				Tools []json.RawMessage `json:"tools"`
			}
			if err := json.Unmarshal(result.Body, &body); err != nil || len(body.Tools) != tc.tools {
				t.Fatalf("backend tools = %d, want %d (error %v): %s", len(body.Tools), tc.tools, err, result.Body)
			}
			var reported int
			for _, diagnostic := range result.Diagnostics {
				if diagnostic.Field == llmprotocol.LossyFieldToolDeferLoading {
					reported++
					if diagnostic.Action != llmprotocol.DiagnosticDropped {
						t.Fatalf("diagnostic action = %q, want dropped", diagnostic.Action)
					}
				}
			}
			if reported != 1 {
				t.Fatalf("defer_loading diagnostics = %d, want 1: %+v", reported, result.Diagnostics)
			}
		})
	}
}

// Once tool search runs, Claude Code returns tool_reference blocks in the
// search's tool result. Under the same opt-in they reach the backend as the
// tool's name, since the tool itself is already sent in full.
func TestDeferLoadingOptInCarriesToolReferences(t *testing.T) {
	body := []byte(`{"model":"m","max_tokens":64,"messages":[
		{"role":"user","content":"fetch it"},
		{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"ToolSearch","input":{"query":"select:WebFetch"}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"tool_reference","tool_name":"WebFetch"}]},{"type":"text","text":"Tool loaded."}]}],
		"tools":[{"name":"ToolSearch","input_schema":{"type":"object"}},{"name":"WebFetch","input_schema":{"type":"object"},"defer_loading":true}]}`)
	route := func(request *llmprotocol.Request) error { request.Model = "routed"; return nil }
	strict, err := NewEngine(NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	_, err = strict.TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, body, route)
	assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_content")

	policy := llmprotocol.DefaultPolicy()
	policy.AllowedLossyFields = []string{llmprotocol.LossyFieldToolDeferLoading}
	engine, err := NewEngine(NewBuiltinRegistry(), policy)
	if err != nil {
		t.Fatal(err)
	}
	for _, target := range []llmprotocol.WireFormat{llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1} {
		result, err := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, target, body, route)
		if err != nil {
			t.Fatalf("%s: opted-in translation failed: %v", target, err)
		}
		if bytes.Contains(result.Body, []byte("tool_reference")) || !bytes.Contains(result.Body, []byte(`WebFetch`)) {
			t.Fatalf("%s: tool reference not carried as the tool name: %s", target, result.Body)
		}
	}
}
