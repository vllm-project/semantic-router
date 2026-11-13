package extproc

import (
	"context"
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// TestKeywordToolResultInputRoutesDecodedToolResults drives a keyword rule
// that reads only the tool results from each wire format through request
// decoding, tool-result extraction and decision evaluation. The same secret in
// the user prompt must not match, because the rule leaves the user turn out.
func TestKeywordToolResultInputRoutesDecodedToolResults(t *testing.T) {
	// AKIAIOSFODNN7EXAMPLE is the example access key id from the AWS
	// documentation, not a live credential.
	const secret = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
	cfg, err := config.ParseYAMLBytes([]byte(`version: v0.5
providers:
  defaults: {model: model-a}
  models:
    - name: model-a
      backend_refs: [{name: model-a, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
recipes:
  - name: default
    signals:
      keywords:
        - name: credentials
          operator: OR
          method: regex
          case_sensitive: true
          keywords: ['\bAKIA[0-9A-Z]{16}\b']
          input: {tool_results: true, request: false}
    decisions:
      - name: credential-route
        priority: 10
        rules: {type: keyword, name: credentials}
        modelRefs: [{model: model-a, use_reasoning: false}]
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	classifier, err := classification.NewClassifier(cfg)
	if err != nil {
		t.Fatalf("NewClassifier() error = %v", err)
	}
	router := &OpenAIRouter{Config: cfg, Classifier: classifier}
	engine := protocolcodec.NewBuiltinEngine()

	toolResultBodies := map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIChatV1: `{"model":"vllm-sr/auto","messages":[` +
			`{"role":"user","content":"read the config"},` +
			`{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{}"}}]},` +
			`{"role":"tool","tool_call_id":"call_1","content":%q}]}`,
		llmprotocol.AnthropicMessagesV1: `{"model":"vllm-sr/auto","max_tokens":64,"messages":[` +
			`{"role":"user","content":"read the config"},` +
			`{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"read_file","input":{}}]},` +
			`{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":%q}]}]}`,
		llmprotocol.OpenAIResponsesV1: `{"model":"vllm-sr/auto","input":[` +
			`{"role":"user","content":"read the config"},` +
			`{"type":"function_call","call_id":"call_1","name":"read_file","arguments":"{}"},` +
			`{"type":"function_call_output","call_id":"call_1","output":%q}]}`,
	}
	promptBodies := map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIChatV1:        `{"model":"vllm-sr/auto","messages":[{"role":"user","content":%q}]}`,
		llmprotocol.AnthropicMessagesV1: `{"model":"vllm-sr/auto","max_tokens":64,"messages":[{"role":"user","content":%q}]}`,
		llmprotocol.OpenAIResponsesV1:   `{"model":"vllm-sr/auto","input":[{"role":"user","content":%q}]}`,
	}

	decide := func(t *testing.T, format llmprotocol.WireFormat, body string) string {
		t.Helper()
		request, _, _, err := engine.DecodeRequest(format, []byte(fmt.Sprintf(body, secret)))
		if err != nil {
			t.Fatalf("DecodeRequest(%s) error = %v", format, err)
		}
		history := extractSignalConversationHistory(&request)
		decision, _, _, _, err := router.performDecisionEvaluation(
			"vllm-sr/auto",
			history,
			&RequestContext{TraceContext: context.Background(), Headers: map[string]string{}},
		)
		if err != nil {
			t.Fatalf("performDecisionEvaluation(%s) error = %v", format, err)
		}
		return decision
	}

	for format, body := range toolResultBodies {
		t.Run(string(format), func(t *testing.T) {
			if decision := decide(t, format, body); decision != "credential-route" {
				t.Fatalf("tool result carrying the key: decision = %q, want credential-route", decision)
			}
			if decision := decide(t, format, promptBodies[format]); decision == "credential-route" {
				t.Fatalf("key in the user prompt: decision = %q, want no credential-route match", decision)
			}
		})
	}
}
