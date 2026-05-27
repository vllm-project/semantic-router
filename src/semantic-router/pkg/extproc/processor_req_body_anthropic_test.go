package extproc

import (
	"encoding/json"
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
