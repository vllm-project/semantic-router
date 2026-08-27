package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestHandleAnthropicRoutingStartsRouterReplay(t *testing.T) {
	cfg := &config.RouterConfig{
		RouterReplay: config.RouterReplayConfig{Enabled: true, StoreBackend: "memory"},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{Name: "simple-queries", ModelRefs: []config.ModelRef{{Model: "claude-sonnet-4.6"}}},
			},
		},
	}
	recorders := initializeReplayRecorders(cfg)
	router := &OpenAIRouter{
		Config:          cfg,
		ReplayRecorders: recorders,
		CredentialResolver: authz.NewCredentialResolver(
			authz.NewHeaderInjectionProvider(map[string]string{
				string(authz.ProviderAnthropic): "x-user-anthropic-key",
			}),
		),
	}
	router.CredentialResolver.SetFailOpen(true)

	request, err := parseOpenAIRequest([]byte(`{"model":"MoM","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("parseOpenAIRequest failed: %v", err)
	}

	ctx := &RequestContext{
		Headers:                  map[string]string{},
		RouterReplayPluginConfig: cfg.EffectiveRouterReplayConfigForDecision("simple-queries"),
		OriginalRequestBody:      []byte(`{"model":"MoM","messages":[{"role":"user","content":"hi"}]}`),
	}
	response, err := router.handleAnthropicRouting(request, "MoM", "claude-sonnet-4.6", "simple-queries", ctx)
	if err != nil {
		t.Fatalf("handleAnthropicRouting failed: %v", err)
	}
	if response == nil {
		t.Fatal("expected routing response")
	}
	if ctx.RouterReplayID == "" {
		t.Fatal("expected router replay to start on anthropic routing path")
	}
}

func TestHandleAnthropicRouting_AllowsStreaming(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{},
		CredentialResolver: authz.NewCredentialResolver(
			authz.NewHeaderInjectionProvider(map[string]string{
				string(authz.ProviderAnthropic): "x-user-anthropic-key",
			}),
		),
	}
	router.CredentialResolver.SetFailOpen(true)

	request, err := parseOpenAIRequest([]byte(`{"model":"claude","messages":[{"role":"user","content":"hi"}],"stream":true}`))
	if err != nil {
		t.Fatalf("parseOpenAIRequest failed: %v", err)
	}

	ctx := &RequestContext{
		Headers:                 map[string]string{},
		ExpectStreamingResponse: true,
		OriginalRequestBody:     []byte(`{"model":"claude","messages":[{"role":"user","content":"hi"}],"stream":true}`),
	}
	response, err := router.handleAnthropicRouting(request, "claude", "claude-sonnet-4.6", "", ctx)
	if err != nil {
		t.Fatalf("handleAnthropicRouting failed: %v", err)
	}
	if response == nil {
		t.Fatal("expected routing response for streaming anthropic request")
	}
	if ctx.AnthropicStream == nil {
		t.Fatal("expected anthropic stream state to be initialized")
	}
	body := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	if !containsJSONField(t, body, "stream", true) {
		t.Fatalf("expected stream=true in anthropic body, got %s", string(body))
	}
}

func TestAnthropicRoutingUsesCompressedToolResultPassthrough(t *testing.T) {
	largeTool := strings.Repeat("irrelevant logs ", 300) +
		"authentication validator failed " +
		strings.Repeat("billing records ", 300)
	body := []byte(`{
		"model":"claude",
		"max_tokens":256,
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
	router := &OpenAIRouter{
		Config: &config.RouterConfig{},
		CredentialResolver: authz.NewCredentialResolver(
			authz.NewHeaderInjectionProvider(map[string]string{
				string(authz.ProviderAnthropic): "x-user-anthropic-key",
			}),
		),
	}
	router.CredentialResolver.SetFailOpen(true)
	ctx := &RequestContext{
		Headers:             map[string]string{},
		ClientProtocol:      config.ClientProtocolAnthropic,
		OriginalRequestBody: body,
		VSRSelectedDecision: contextCompressionTestDecision(false),
	}

	request, early, err := router.prepareRequestForModelRouting(
		body,
		"fix authentication validator",
		ctx,
	)
	if err != nil || early != nil {
		t.Fatalf("prepare request failed: early=%v err=%v", early, err)
	}
	_, outbound, errorResponse := router.prepareAnthropicRoutingRequest(request, "claude", ctx)
	if errorResponse != nil {
		t.Fatalf("prepare Anthropic routing returned error response: %#v", errorResponse)
	}

	var decoded map[string]interface{}
	if err := json.Unmarshal(outbound, &decoded); err != nil {
		t.Fatalf("outbound Anthropic body is invalid: %v", err)
	}
	toolResult := findAnthropicToolResult(decoded)
	if toolResult == nil {
		t.Fatalf("outbound body lost tool_result: %s", outbound)
	}
	if toolResult["is_error"] != true || toolResult["tool_use_id"] != "call_1" {
		t.Fatalf("outbound metadata changed: %#v", toolResult)
	}
	resultBlocks := toolResult["content"].([]interface{})
	compressedText := resultBlocks[0].(map[string]interface{})["text"].(string)
	if len(compressedText) >= len(largeTool) {
		t.Fatal("passthrough restored stale uncompressed tool result")
	}
	if resultBlocks[1].(map[string]interface{})["type"] != "image" {
		t.Fatalf("outbound image block changed: %#v", resultBlocks[1])
	}
}

func findAnthropicToolResult(body map[string]interface{}) map[string]interface{} {
	messages, _ := body["messages"].([]interface{})
	for _, rawMessage := range messages {
		message, _ := rawMessage.(map[string]interface{})
		content, _ := message["content"].([]interface{})
		for _, rawBlock := range content {
			block, _ := rawBlock.(map[string]interface{})
			if block["type"] == "tool_result" {
				return block
			}
		}
	}
	return nil
}

// TestParseRequestForProtocol_OpenAIDefault verifies the dispatch keeps
// the OpenAI fast path byte-identical when ClientProtocol is empty.
func TestParseRequestForProtocol_OpenAIDefault(t *testing.T) {
	ctx := &RequestContext{Headers: map[string]string{}}
	body := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)
	req, err := parseRequestForProtocol(ctx, body)
	if err != nil {
		t.Fatalf("parseRequestForProtocol: %v", err)
	}
	if req == nil || req.Model != "gpt-4" {
		t.Fatalf("unexpected req: %+v", req)
	}
	if ctx.IRExtensions != nil {
		t.Fatalf("expected nil IRExtensions for OpenAI, got %+v", ctx.IRExtensions)
	}
}

// TestParseRequestForProtocol_AnthropicSetsIRExtensions verifies that an
// Anthropic-shape body routes through ParseAnthropicRequest and the
// resulting IRExtensions is stashed on the context for downstream
// emitters and plugins.
func TestParseRequestForProtocol_AnthropicSetsIRExtensions(t *testing.T) {
	ctx := &RequestContext{
		Headers:        map[string]string{":path": "/v1/messages"},
		ClientProtocol: config.ClientProtocolAnthropic,
	}
	body := []byte(`{
		"model": "claude-opus-4-7",
		"max_tokens": 2048,
		"system": [{"type": "text", "text": "be precise", "cache_control": {"type": "ephemeral", "ttl": "5m"}}],
		"top_k": 40,
		"metadata": {"user_id": "user-abc"},
		"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
	}`)
	req, err := parseRequestForProtocol(ctx, body)
	if err != nil {
		t.Fatalf("parseRequestForProtocol: %v", err)
	}
	if req == nil || req.Model != "claude-opus-4-7" {
		t.Fatalf("unexpected req: %+v", req)
	}
	if req.MaxTokens.Value != 2048 {
		t.Fatalf("max_tokens: got %d", req.MaxTokens.Value)
	}
	if req.User.Value != "user-abc" {
		t.Fatalf("user: got %q", req.User.Value)
	}
	if ctx.IRExtensions == nil {
		t.Fatal("expected ctx.IRExtensions to be populated")
	}
	if ctx.IRExtensions.SourceProtocol != "anthropic" {
		t.Fatalf("source protocol: got %q", ctx.IRExtensions.SourceProtocol)
	}
	if ctx.IRExtensions.MetadataUserID != "user-abc" {
		t.Fatalf("metadata user id: got %q", ctx.IRExtensions.MetadataUserID)
	}
	if _, ok := ctx.IRExtensions.CacheControl["system.0"]; !ok {
		t.Fatalf("expected cache control on system.0, got %+v", ctx.IRExtensions.CacheControl)
	}
	if ctx.IRExtensions.TopK == nil || *ctx.IRExtensions.TopK != 40 {
		t.Fatalf("top_k: got %+v", ctx.IRExtensions.TopK)
	}
}

// TestParseRequestForProtocol_AnthropicRejectsInvalidBody confirms that
// parse failures surface as Go errors and do not partially populate
// IRExtensions on the context.
func TestParseRequestForProtocol_AnthropicRejectsInvalidBody(t *testing.T) {
	ctx := &RequestContext{
		Headers:        map[string]string{":path": "/v1/messages"},
		ClientProtocol: config.ClientProtocolAnthropic,
	}
	if _, err := parseRequestForProtocol(ctx, []byte("{not-json")); err == nil {
		t.Fatal("expected error for invalid JSON")
	}
	if ctx.IRExtensions != nil {
		t.Fatalf("expected no IRExtensions on parse failure, got %+v", ctx.IRExtensions)
	}
}

// TestValidateAnthropicRequestBody covers the protocol-keyed validation
// branch added to validateRequestBody. The router error response is the
// authoritative gate that an OpenAI-shape body validator would otherwise
// have rejected an Anthropic body for.
func TestValidateAnthropicRequestBody(t *testing.T) {
	router := &OpenAIRouter{}
	tests := []struct {
		name      string
		body      string
		wantError bool
	}{
		{name: "valid minimal", body: `{"model":"claude","messages":[{"role":"user","content":"hi"}]}`},
		{name: "valid with system array", body: `{"model":"claude","system":[{"type":"text","text":"s"}],"messages":[{"role":"user","content":"hi"}]}`},
		{name: "missing model", body: `{"messages":[{"role":"user","content":"hi"}]}`, wantError: true},
		{name: "empty model", body: `{"model":"","messages":[{"role":"user","content":"hi"}]}`, wantError: true},
		{name: "missing messages", body: `{"model":"claude"}`, wantError: true},
		{name: "messages not array", body: `{"model":"claude","messages":"hi"}`, wantError: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := router.validateAnthropicRequestBody([]byte(tt.body))
			if tt.wantError && resp == nil {
				t.Fatalf("expected error response, got nil")
			}
			if !tt.wantError && resp != nil {
				t.Fatalf("expected nil response, got %+v", resp)
			}
		})
	}
}

// TestValidateRequestBody_AnthropicRoutesThroughAnthropicValidator
// asserts that the protocol-keyed branch is taken when ClientProtocol is
// "anthropic" — without the branch, validateRequestBody short-circuits
// on the /v1/chat/completions path check and accepts ill-formed bodies.
func TestValidateRequestBody_AnthropicRoutesThroughAnthropicValidator(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		Headers:        map[string]string{":path": "/v1/messages"},
		ClientProtocol: config.ClientProtocolAnthropic,
	}
	resp := router.validateRequestBody([]byte(`{"messages":[{"role":"user","content":"hi"}]}`), ctx)
	if resp == nil {
		t.Fatal("expected validation error for missing model")
	}
}

func containsJSONField(t *testing.T, body []byte, key string, want bool) bool {
	t.Helper()
	var parsed map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	got, ok := parsed[key].(bool)
	return ok && got == want
}

// TestAnthropicRoutingRewritesProviderModelID guards #3064: the Anthropic
// routing path must send the backend the model name from
// provider_model_id/external_model_ids, not the router alias, mirroring the
// OpenAI-compatible path.
func TestAnthropicRoutingRewritesProviderModelID(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.ModelConfig = map[string]config.ModelParams{
		"mymodel-code": {
			ExternalModelIDs: map[string]string{
				"vllm": "Qwen/Qwen3.6-35B-A3B-FP8",
			},
		},
	}
	router := &OpenAIRouter{
		Config: cfg,
		CredentialResolver: authz.NewCredentialResolver(
			authz.NewHeaderInjectionProvider(map[string]string{
				string(authz.ProviderAnthropic): "x-user-anthropic-key",
			}),
		),
	}
	router.CredentialResolver.SetFailOpen(true)

	body := []byte(`{"model":"mymodel-code","messages":[{"role":"user","content":"hi"}]}`)
	request, err := parseOpenAIRequest(body)
	if err != nil {
		t.Fatalf("parseOpenAIRequest failed: %v", err)
	}
	ctx := &RequestContext{
		Headers:             map[string]string{},
		OriginalRequestBody: body,
	}

	response, err := router.handleAnthropicRouting(request, "mymodel-code", "mymodel-code", "", ctx)
	if err != nil {
		t.Fatalf("handleAnthropicRouting failed: %v", err)
	}
	outbound := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	if len(outbound) == 0 {
		t.Fatalf("expected outbound body mutation, got %#v", response)
	}

	var decoded map[string]interface{}
	if err := json.Unmarshal(outbound, &decoded); err != nil {
		t.Fatalf("outbound Anthropic body is invalid: %v", err)
	}
	if got := decoded["model"]; got != "Qwen/Qwen3.6-35B-A3B-FP8" {
		t.Fatalf("outbound model = %v, want provider model ID", got)
	}
}

// TestAnthropicRoutingKeepsAliasWithoutExternalModelIDs is the control for
// #3064: with no external_model_ids mapping the outbound model is unchanged.
func TestAnthropicRoutingKeepsAliasWithoutExternalModelIDs(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{},
		CredentialResolver: authz.NewCredentialResolver(
			authz.NewHeaderInjectionProvider(map[string]string{
				string(authz.ProviderAnthropic): "x-user-anthropic-key",
			}),
		),
	}
	router.CredentialResolver.SetFailOpen(true)

	body := []byte(`{"model":"claude","messages":[{"role":"user","content":"hi"}]}`)
	request, err := parseOpenAIRequest(body)
	if err != nil {
		t.Fatalf("parseOpenAIRequest failed: %v", err)
	}
	ctx := &RequestContext{
		Headers:             map[string]string{},
		OriginalRequestBody: body,
	}

	response, err := router.handleAnthropicRouting(request, "claude", "claude-sonnet-4.6", "", ctx)
	if err != nil {
		t.Fatalf("handleAnthropicRouting failed: %v", err)
	}
	outbound := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	var decoded map[string]interface{}
	if err := json.Unmarshal(outbound, &decoded); err != nil {
		t.Fatalf("outbound Anthropic body is invalid: %v", err)
	}
	if got := decoded["model"]; got != "claude-sonnet-4.6" {
		t.Fatalf("outbound model = %v, want claude-sonnet-4.6", got)
	}
}
