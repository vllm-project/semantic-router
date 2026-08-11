package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
)

func TestApplyContextCompressionCompressesOnlyLargeToolOutput(t *testing.T) {
	decision := &config.Decision{
		Name: "compressed-route",
		Plugins: []config.DecisionPlugin{{
			Type: "context_compression",
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"targets": map[string]interface{}{
					"tool_outputs": map[string]interface{}{
						"mode":          "extractive",
						"min_tokens":    100,
						"target_tokens": 45,
					},
				},
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

func TestApplyContextCompressionHonorsRequestControl(t *testing.T) {
	decision := &config.Decision{
		Name: "compressed-route",
		Plugins: []config.DecisionPlugin{{
			Type: "context_compression",
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"targets": map[string]interface{}{
					"tool_outputs": map[string]interface{}{
						"mode":          "extractive",
						"min_tokens":    10,
						"target_tokens": 5,
					},
				},
				"request_controls": map[string]interface{}{
					"enabled": true,
					"header":  "x-compression-control",
					"allowed": []string{"bypass"},
				},
			}),
		}},
	}
	body := []byte(`{"model":"auto","messages":[{"role":"tool","content":"one two three four five six seven eight nine ten eleven twelve"}]}`)
	ctx := &RequestContext{
		Headers:             map[string]string{"x-compression-control": "bypass"},
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

func TestApplyContextCompressionPreservesJSONAndArrayBlocks(t *testing.T) {
	decision := contextCompressionTestDecision(false)
	jsonTool := `{"status":"authentication validator failed","logs":"` +
		strings.Repeat("x", 5000) +
		`","nested":{"code":500}}`
	request := map[string]interface{}{
		"model": "auto",
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "fix authentication validator"},
			map[string]interface{}{
				"role": "tool",
				"content": []interface{}{
					map[string]interface{}{"type": "text", "text": jsonTool},
					map[string]interface{}{"type": "image_url", "image_url": map[string]interface{}{"url": "https://example.com/a.png"}},
				},
			},
		},
	}
	body, err := json.Marshal(request)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	ctx := &RequestContext{VSRSelectedDecision: decision}

	compressed := (&OpenAIRouter{}).applyContextCompression(ctx, body)

	var decoded map[string]interface{}
	if err := json.Unmarshal(compressed, &decoded); err != nil {
		t.Fatalf("compressed request is invalid JSON: %v", err)
	}
	messages := decoded["messages"].([]interface{})
	tool := messages[1].(map[string]interface{})
	blocks := tool["content"].([]interface{})
	text := blocks[0].(map[string]interface{})["text"].(string)
	if !json.Valid([]byte(text)) {
		t.Fatalf("JSON tool text was corrupted: %s", text)
	}
	if blocks[1].(map[string]interface{})["type"] != "image_url" {
		t.Fatalf("non-text content block changed: %#v", blocks[1])
	}
	if ctx.ContextCompressionFormat != "json" {
		t.Fatalf("expected JSON diagnostics, got %q", ctx.ContextCompressionFormat)
	}
}

func TestApplyContextCompressionHandlesAnthropicToolResultArray(t *testing.T) {
	decision := contextCompressionTestDecision(false)
	largeTool := strings.Repeat("irrelevant logs ", 300) +
		"authentication validator failed " +
		strings.Repeat("billing records ", 300)
	body := []byte(`{
		"model":"claude",
		"messages":[{
			"role":"user",
			"content":[
				{"type":"text","text":"fix authentication validator"},
				{"type":"tool_result","tool_use_id":"call_1","is_error":true,"content":[
					{"type":"text","text":` + mustJSONString(t, largeTool) + `},
					{"type":"image","source":{"type":"base64","media_type":"image/png","data":"abc"}}
				]}
			]
		}]
	}`)
	ctx := &RequestContext{
		ClientProtocol:      config.ClientProtocolAnthropic,
		VSRSelectedDecision: decision,
	}

	compressed := (&OpenAIRouter{}).applyContextCompression(ctx, body)

	var decoded map[string]interface{}
	if err := json.Unmarshal(compressed, &decoded); err != nil {
		t.Fatalf("compressed Anthropic request is invalid: %v", err)
	}
	messages := decoded["messages"].([]interface{})
	content := messages[0].(map[string]interface{})["content"].([]interface{})
	toolResult := content[1].(map[string]interface{})
	if toolResult["is_error"] != true || toolResult["tool_use_id"] != "call_1" {
		t.Fatalf("Anthropic tool-result metadata changed: %#v", toolResult)
	}
	resultBlocks := toolResult["content"].([]interface{})
	if resultBlocks[1].(map[string]interface{})["type"] != "image" {
		t.Fatalf("Anthropic image block changed: %#v", resultBlocks[1])
	}
	compressedText := resultBlocks[0].(map[string]interface{})["text"].(string)
	if len(compressedText) >= len(largeTool) {
		t.Fatal("Anthropic text block was not compressed")
	}
}

func TestApplyContextCompressionProtectsRAGUnlessOptedIn(t *testing.T) {
	largeTool := strings.Repeat("retrieved authentication context ", 300)
	request := map[string]interface{}{
		"model": "auto",
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "authentication"},
			map[string]interface{}{
				"role":         "tool",
				"tool_call_id": "rag_req_1",
				"content":      largeTool,
			},
		},
	}
	body, err := json.Marshal(request)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	protectedCtx := &RequestContext{VSRSelectedDecision: contextCompressionTestDecision(false)}
	if got := (&OpenAIRouter{}).applyContextCompression(protectedCtx, body); string(got) != string(body) {
		t.Fatal("RAG tool result changed without explicit opt-in")
	}
	if protectedCtx.ContextCompressionSkipReason != "rag_protected" {
		t.Fatalf("missing RAG protection diagnostic: %#v", protectedCtx)
	}

	optedInCtx := &RequestContext{VSRSelectedDecision: contextCompressionTestDecision(true)}
	if got := (&OpenAIRouter{}).applyContextCompression(optedInCtx, body); string(got) == string(body) {
		t.Fatal("RAG tool result was not compressed after opt-in")
	}
}

func TestPrepareRequestForModelRoutingReparsesCompressedWorkingBody(t *testing.T) {
	decision := contextCompressionTestDecision(false)
	largeTool := strings.Repeat("irrelevant inventory ", 300) +
		"authentication validator failed " +
		strings.Repeat("irrelevant billing ", 300)
	body := []byte(`{"model":"auto","messages":[` +
		`{"role":"user","content":"fix authentication validator"},` +
		`{"role":"tool","tool_call_id":"call_1","content":` + mustJSONString(t, largeTool) + `}]}`)
	original := append([]byte(nil), body...)
	ctx := &RequestContext{
		Headers:             map[string]string{},
		OriginalRequestBody: original,
		VSRSelectedDecision: decision,
	}
	router := &OpenAIRouter{Config: &config.RouterConfig{}}

	parsed, early, err := router.prepareRequestForModelRouting(body, "fix authentication validator", ctx)

	if err != nil || early != nil {
		t.Fatalf("prepare request failed: early=%v err=%v", early, err)
	}
	if string(ctx.OriginalRequestBody) != string(original) {
		t.Fatal("immutable original request body changed")
	}
	if !ctx.requestBodyMutated() {
		t.Fatal("working request body did not record compression")
	}
	serialized, err := serializeOpenAIRequestWithStream(parsed, false)
	if err != nil {
		t.Fatalf("serialize parsed request: %v", err)
	}
	if strings.Contains(string(serialized), strings.Repeat("irrelevant inventory ", 30)) {
		t.Fatal("provider IR was not reparsed from compressed working body")
	}
}

func TestApplyContextCompressionMalformedBodyFailsOpen(t *testing.T) {
	body := []byte(`{"messages":[`)
	ctx := &RequestContext{VSRSelectedDecision: contextCompressionTestDecision(false)}

	got := (&OpenAIRouter{}).applyContextCompression(ctx, body)

	if string(got) != string(body) {
		t.Fatalf("malformed body changed: %q", got)
	}
	if ctx.ContextCompressionSkipReason != "parse_error_fail_open" {
		t.Fatalf("missing fail-open diagnostic: %#v", ctx)
	}
}

func TestContextCompressionPreservesRecoverableTargetsForStreaming(t *testing.T) {
	decision := &config.Decision{
		Name: "streaming-recovery",
		Plugins: []config.DecisionPlugin{{
			Type: config.DecisionPluginContextCompression,
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"mode":    "always",
				"targets": map[string]interface{}{
					"tool_outputs": map[string]interface{}{
						"mode":          "recoverable",
						"min_tokens":    10,
						"target_tokens": 5,
					},
				},
				"recovery": map[string]interface{}{
					"enabled": true,
					"store":   "redis",
				},
			}),
		}},
	}
	body := []byte(`{"messages":[{"role":"user","content":"question"},{"role":"tool","content":"` +
		strings.Repeat("large output ", 50) + `"}]}`)
	ctx := &RequestContext{
		VSRSelectedDecision:     decision,
		ExpectStreamingResponse: true,
	}
	got := (&OpenAIRouter{}).applyContextCompression(ctx, body)
	if string(got) != string(body) {
		t.Fatal("streaming request exposed recoverable compression")
	}
}

func TestContextCompressionFailClosedReturnsError(t *testing.T) {
	decision := &config.Decision{
		Name: "fail-closed",
		Plugins: []config.DecisionPlugin{{
			Type: config.DecisionPluginContextCompression,
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled":      true,
				"failure_mode": "fail_closed",
			}),
		}},
	}
	_, err := (&OpenAIRouter{}).applyContextCompressionPolicy(
		&RequestContext{VSRSelectedDecision: decision},
		[]byte(`not-json`),
	)
	if err == nil {
		t.Fatal("fail_closed compression accepted malformed request")
	}
}

func TestInjectContextRecoveryToolRejectsReservedNameConflict(t *testing.T) {
	request := map[string]interface{}{
		"tools": []interface{}{map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name": contextcompression.RetrieveToolName,
			},
		}},
	}
	if err := injectContextRecoveryTool(request, []string{"issued-key"}); err == nil {
		t.Fatal("reserved recovery tool conflict was accepted")
	}
}

func contextCompressionTestDecision(compressRAG bool) *config.Decision {
	ragMode := "preserve"
	if compressRAG {
		ragMode = "extractive"
	}
	return &config.Decision{
		Name: "compressed-route",
		Plugins: []config.DecisionPlugin{{
			Type: "context_compression",
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"targets": map[string]interface{}{
					"tool_outputs": map[string]interface{}{
						"mode":          "extractive",
						"min_tokens":    200,
						"target_tokens": 160,
					},
					"rag": map[string]interface{}{
						"mode": ragMode,
					},
				},
			}),
		}},
	}
}

func mustJSONString(t *testing.T, value string) string {
	t.Helper()
	body, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("marshal string: %v", err)
	}
	return string(body)
}
