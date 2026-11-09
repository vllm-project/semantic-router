package extproc

import (
	"context"
	"fmt"
	"strings"
	"testing"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A gateway that serves Claude in the Anthropic Messages format adds its own
// cost to usage. No backend setting is involved anywhere below.
const anthropicGatewayReply = `{"id":"msg_1","type":"message","role":"assistant","model":"provider-model",` +
	`"content":[{"type":"text","text":"done"}],"stop_reason":"end_turn","stop_sequence":null,` +
	`"usage":{"input_tokens":12,"output_tokens":4,"speed":"standard","total_cost":3e-05}}`

func anthropicGatewayContext(source llmprotocol.WireFormat) *RequestContext {
	return &RequestContext{
		RequestID:    "req-gateway",
		SourceFormat: source,
		TargetFormat: llmprotocol.AnthropicMessagesV1,
		RequestModel: "public-model",
		TraceContext: context.Background(),
	}
}

// The buffered reply decodes with its usage, a same-format client gets the
// upstream bytes, and the bytes a cache stores are canonical.
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
	if !ctx.ResponseUnknownFields {
		t.Fatal("ResponseUnknownFields = false, want true for a reply carrying gateway fields")
	}
	if requiresClientResponseRewrite(ctx) || ctx.ResponseBodyNeedsRewrite {
		t.Fatal("a same-format reply is rewritten, so the client would lose the gateway's fields")
	}
	cached := router.cacheableClientResponse([]byte(anthropicGatewayReply), false, *response, ctx)
	if len(cached) == 0 || strings.Contains(string(cached), "total_cost") || !strings.Contains(string(cached), `"text":"done"`) {
		t.Fatalf("cached body = %s, want the canonical encode without gateway fields", cached)
	}
	hitCtx := anthropicGatewayContext(llmprotocol.AnthropicMessagesV1)
	if _, hitErr := router.decodeCachedClientResponse(cached, hitCtx); hitErr != nil {
		t.Fatalf("decodeCachedClientResponse() error = %v on the cached body", hitErr)
	}

	chatCtx := anthropicGatewayContext(llmprotocol.OpenAIChatV1)
	chat, err := router.decodeClientResponse([]byte(anthropicGatewayReply), chatCtx)
	if err != nil {
		t.Fatalf("decodeClientResponse(Chat client) error = %v", err)
	}
	body, err := router.encodeClientResponse(*chat, chatCtx)
	if err != nil {
		t.Fatalf("encodeClientResponse(Chat client) error = %v", err)
	}
	if strings.Contains(string(body), "total_cost") || !strings.Contains(string(body), `"prompt_tokens":12`) {
		t.Fatalf("Chat client body = %s", body)
	}
}

func TestAnthropicGatewayStreamCompletes(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := anthropicGatewayContext(llmprotocol.AnthropicMessagesV1)
	body := []byte(strings.Join([]string{
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":12,\"output_tokens\":1,\"speed\":\"standard\",\"total_cost\":1e-06}}}\n\n",
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

// Native fallback re-dispatches outside Envoy's filter chain, so it decodes
// the candidate's reply under the same contract as a first attempt.
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
					Type: "anthropic-compatible", Protocol: "anthropic/messages@1",
					BaseURL: "http://127.0.0.1:8002",
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
	if !ctx.ResponseUnknownFields {
		t.Fatal("ResponseUnknownFields = false, want the drop recorded")
	}
	if body := string(resp.GetImmediateResponse().GetBody()); strings.Contains(body, "total_cost") ||
		!strings.Contains(body, `"prompt_tokens":12`) {
		t.Fatalf("Chat client body = %s", body)
	}
}

// A gzip reply is decoded before the codec reads it, so a compressed gateway
// reply needs no setting either, through the full response phases.
func TestGzipAnthropicGatewayReplyDecodes(t *testing.T) {
	for _, client := range []llmprotocol.WireFormat{llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1} {
		t.Run(string(client), func(t *testing.T) {
			ctx := looperTransportContext(llmprotocol.AnthropicMessagesV1, false)
			ctx.LooperRequest = false
			ctx.SourceFormat = client
			result := runBufferedReply(t, &OpenAIRouter{}, ctx, 200, gzipBytes(t, []byte(anthropicGatewayReply)), "gzip")

			require.Nil(t, result.body.GetImmediateResponse(), "gzip gateway reply was rejected: %v", result.body)
			require.True(t, ctx.ResponseUnknownFields)
			if client == llmprotocol.AnthropicMessagesV1 {
				require.Contains(t, string(result.clientBody), `"total_cost":3e-05`)
				return
			}
			require.NotContains(t, string(result.clientBody), "total_cost")
			require.Contains(t, string(result.clientBody), `"prompt_tokens":12`)
		})
	}
}

// An OpenAI-format relay names its usage counters the Anthropic way and adds a
// cost. A Chat client receives the relay's bytes; a Messages client receives
// the counters it reported.
const chatRelayReply = `{"id":"chatcmpl-relay","object":"chat.completion","created":1,"model":"relay-model",` +
	`"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"done"}}],` +
	`"usage":{"input_tokens":12,"output_tokens":4,"total_tokens":16,"total_cost":3e-05}}`

func TestChatRelayReplyNeedsNoSetting(t *testing.T) {
	for _, client := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1} {
		t.Run(string(client), func(t *testing.T) {
			ctx := looperTransportContext(llmprotocol.OpenAIChatV1, false)
			ctx.LooperRequest = false
			ctx.SourceFormat = client
			result := runBufferedReply(t, &OpenAIRouter{}, ctx, 200, []byte(chatRelayReply), "")

			require.Nil(t, result.body.GetImmediateResponse(), "relay reply was rejected: %v", result.body)
			require.True(t, ctx.ResponseUnknownFields)
			require.NotNil(t, ctx.SemanticResponse)
			require.EqualValues(t, 12, *ctx.SemanticResponse.Usage.InputTotal.Value)
			require.EqualValues(t, 4, *ctx.SemanticResponse.Usage.OutputTotal.Value)
			if client == llmprotocol.OpenAIChatV1 {
				require.Contains(t, string(result.clientBody), `"total_cost":3e-05`)
				require.Contains(t, string(result.clientBody), `"input_tokens":12`)
				return
			}
			require.NotContains(t, string(result.clientBody), "total_cost")
			require.Contains(t, string(result.clientBody), `"input_tokens":12`)
		})
	}
}
