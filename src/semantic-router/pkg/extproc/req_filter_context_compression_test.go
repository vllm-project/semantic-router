package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestApplyContextCompressionCompressesOnlyLargeToolOutput(t *testing.T) {
	decision := &config.Decision{
		Name: "compressed-route",
		Plugins: []config.DecisionPlugin{{
			Type: "context_compression",
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled":       true,
				"min_tokens":    100,
				"target_tokens": 45,
			}),
		}},
	}
	largeTool := strings.Join([]string{
		"source header",
		strings.Repeat("irrelevant inventory values ", 120),
		"authentication token validator failed",
		strings.Repeat("irrelevant billing values ", 120),
		"source footer",
	}, "\n")
	request := map[string]interface{}{
		"model": "auto",
		"messages": []map[string]interface{}{
			{"role": "user", "content": "fix authentication token validation"},
			{"role": "tool", "content": largeTool, "tool_call_id": "call-1"},
		},
	}
	body, err := json.Marshal(request)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	ctx := &RequestContext{
		RequestID:           "req-compress",
		UserContent:         "fix authentication token validation",
		VSRSelectedDecision: decision,
	}
	router := &OpenAIRouter{}

	compressed := router.applyContextCompression(ctx, body)

	if string(compressed) == string(body) {
		t.Fatal("large tool output was not compressed")
	}
	if !ctx.ContextCompressionApplied || ctx.ContextCompressionMessages != 1 {
		t.Fatalf("compression diagnostics missing: %#v", ctx)
	}
	if ctx.ContextCompressionAfter >= ctx.ContextCompressionBefore {
		t.Fatalf("token count did not decrease: before=%d after=%d", ctx.ContextCompressionBefore, ctx.ContextCompressionAfter)
	}
	if !strings.Contains(string(compressed), "authentication token validator") {
		t.Fatalf("query-relevant tool content was removed: %s", compressed)
	}
	if !strings.Contains(string(compressed), `"role":"user"`) {
		t.Fatalf("user message shape changed: %s", compressed)
	}
}

func TestApplyContextCompressionHonorsConfiguredBypassHeader(t *testing.T) {
	decision := &config.Decision{
		Name: "compressed-route",
		Plugins: []config.DecisionPlugin{{
			Type: "context_compression",
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled":       true,
				"min_tokens":    10,
				"target_tokens": 5,
				"bypass_header": "x-compression-bypass",
			}),
		}},
	}
	body := []byte(`{"model":"auto","messages":[{"role":"tool","content":"one two three four five six seven eight nine ten eleven twelve"}]}`)
	ctx := &RequestContext{
		Headers:             map[string]string{"x-compression-bypass": "true"},
		VSRSelectedDecision: decision,
	}
	router := &OpenAIRouter{}

	if got := router.applyContextCompression(ctx, body); string(got) != string(body) {
		t.Fatalf("bypassed request changed: %s", got)
	}
	if ctx.ContextCompressionApplied {
		t.Fatal("bypassed request recorded compression")
	}
}
