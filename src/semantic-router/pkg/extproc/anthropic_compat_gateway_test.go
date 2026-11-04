package extproc

import (
	"context"
	"fmt"
	"strings"
	"testing"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// An enterprise gateway that serves Claude in the Anthropic Messages format
// adds its own cost field to usage.
const anthropicGatewayReply = `{"id":"msg_1","type":"message","role":"assistant","model":"provider-model",` +
	`"content":[{"type":"text","text":"done"}],"stop_reason":"end_turn","stop_sequence":null,` +
	`"usage":{"input_tokens":12,"output_tokens":4,"total_cost":3e-05}}`

func anthropicGatewayContext(source llmprotocol.WireFormat) *RequestContext {
	return &RequestContext{
		RequestID:      "req-gateway",
		SourceFormat:   source,
		TargetFormat:   llmprotocol.AnthropicMessagesV1,
		RequestModel:   "public-model",
		ResponseVendor: llmprotocol.ResponseVendorAnthropicCompatGateway,
		TraceContext:   context.Background(),
	}
}

// The buffered reply decodes with its usage, and the bytes a cache stores are
// readable by the strict cache reader.
func TestAnthropicGatewayReplyDecodesAndCachesCanonically(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := anthropicGatewayContext(llmprotocol.AnthropicMessagesV1)
	response, err := router.decodeClientResponse([]byte(anthropicGatewayReply), ctx)
	if err != nil {
		t.Fatalf("decodeClientResponse() error = %v", err)
	}
	if response.Usage.InputTotal.Value == nil || *response.Usage.InputTotal.Value != 12 ||
		response.Usage.OutputTotal.Value == nil || *response.Usage.OutputTotal.Value != 4 {
		t.Fatalf("usage = %+v", response.Usage)
	}
	if !ctx.ResponseVendorExtensions {
		t.Fatal("ResponseVendorExtensions = false, want true for a reply carrying gateway fields")
	}
	cached := router.cacheableClientResponse([]byte(anthropicGatewayReply), false, *response, ctx)
	if len(cached) == 0 || strings.Contains(string(cached), "total_cost") {
		t.Fatalf("cached body = %s, want the canonical encode without gateway fields", cached)
	}
	hitCtx := anthropicGatewayContext(llmprotocol.AnthropicMessagesV1)
	hitCtx.ResponseVendor = ""
	if _, err := router.decodeCachedClientResponse(cached, hitCtx); err != nil {
		t.Fatalf("decodeCachedClientResponse() error = %v on the cached body", err)
	}

	strict := anthropicGatewayContext(llmprotocol.OpenAIChatV1)
	strict.ResponseVendor = ""
	if _, err := router.decodeClientResponse([]byte(anthropicGatewayReply), strict); err == nil {
		t.Fatal("a backend without the opt-in accepted gateway fields")
	}
}

func TestAnthropicGatewayStreamCompletes(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := anthropicGatewayContext(llmprotocol.AnthropicMessagesV1)
	body := []byte(strings.Join([]string{
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":12,\"output_tokens\":1,\"total_cost\":1e-06}}}\n\n",
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"done\"}}\n\n",
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":4,\"total_cost\":3e-05}}\n\n",
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	}, ""))
	router.handleSemanticStreamingResponseBody(body, true, ctx)
	if ctx.StreamingAborted || !ctx.StreamingComplete {
		t.Fatalf("gateway stream did not complete: aborted=%v complete=%v", ctx.StreamingAborted, ctx.StreamingComplete)
	}
	if ctx.SemanticResponse == nil || ctx.SemanticResponse.Usage.OutputTotal.Value == nil ||
		*ctx.SemanticResponse.Usage.OutputTotal.Value != 4 {
		t.Fatalf("settled usage changed: %+v", ctx.SemanticResponse)
	}
}

// Native fallback resolves the candidate's response contract from its own
// backend profile, so an Anthropic gateway candidate is decoded under its
// opt-in like a first attempt.
func TestFallbackAnthropicGatewayCandidateDecodesGatewayFields(t *testing.T) {
	policy := fallback.DefaultEnabledPolicy()
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: "model-primary",
			ModelConfig: map[string]config.ModelParams{
				"model-primary": {
					PreferredEndpoints: []string{"backend-primary"},
					APIFormat:          config.APIFormatOpenAI,
				},
				"model-gateway-candidate": {
					PreferredEndpoints: []string{"backend-gateway"},
					APIFormat:          config.APIFormatAnthropic,
				},
			},
			VLLMEndpoints: []config.VLLMEndpoint{
				{Name: "backend-primary", Address: "127.0.0.1", Port: 8001, ProviderProfileName: "prof-openai"},
				{Name: "backend-gateway", Address: "127.0.0.1", Port: 8002, ProviderProfileName: "prof-gateway"},
			},
			ProviderProfiles: map[string]config.ProviderProfile{
				"prof-openai": {Type: "openai", BaseURL: "http://127.0.0.1:8001"},
				"prof-gateway": {
					Type: "anthropic", Protocol: anthropicMessagesProtocol,
					BaseURL: "http://127.0.0.1:8002", CompatGateway: true,
				},
			},
		},
	}
	orchestrator := fallback.NewOrchestrator(policy, fallback.NewBackendCircuitBreaker(policy.CircuitBreaker))
	router := &OpenAIRouter{
		Config:               cfg,
		CredentialResolver:   newTestCredentialResolver(cfg),
		FallbackOrchestrator: orchestrator,
	}
	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-gateway-candidate"})
	ctx.UpstreamStatusCode = 503
	router.fallbackCaller = func(_ context.Context, model string, _ []byte, _ map[string]string) ([]byte, int, error) {
		if model == "model-gateway-candidate" {
			return []byte(anthropicGatewayReply), 200, nil
		}
		return nil, 500, fmt.Errorf("unexpected model: %s", model)
	}

	resp := router.maybeExecuteFallback(nil, ctx)
	if resp == nil || resp.GetImmediateResponse() == nil {
		t.Fatalf("expected an ImmediateResponse from the gateway candidate, got %#v", resp)
	}
	if code := resp.GetImmediateResponse().GetStatus().GetCode(); code != typev3.StatusCode_OK {
		t.Fatalf("fallback status = %v, want 200: %s", code, resp.GetImmediateResponse().GetBody())
	}
	if ctx.ResponseVendor != llmprotocol.ResponseVendorAnthropicCompatGateway || !ctx.ResponseVendorExtensions {
		t.Fatalf("vendor = %q extensions = %v, want the Anthropic gateway contract with its drop recorded",
			ctx.ResponseVendor, ctx.ResponseVendorExtensions)
	}
	if body := string(resp.GetImmediateResponse().GetBody()); strings.Contains(body, "total_cost") ||
		!strings.Contains(body, `"prompt_tokens":12`) {
		t.Fatalf("Chat client body = %s", body)
	}
}
