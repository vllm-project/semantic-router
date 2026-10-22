package extproc

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

const fallbackCachedAnthropicRequest = `{
	"model":"model-primary","max_tokens":64,
	"messages":[{"role":"user","content":[{"type":"text","text":"hello","cache_control":{"type":"ephemeral"}}]}]
}`

func TestAnthropicCacheWarningFromFailedFallbackDoesNotLeak(t *testing.T) {
	for _, failure := range []string{"upstream", "rate_limit"} {
		t.Run(failure, func(t *testing.T) {
			policy := fallback.DefaultEnabledPolicy()
			policy.MaxAttempts = 3
			router, cfg := setupFallbackTestRouter(t, policy)
			responses := cfg.ModelConfig["model-fallback-1"]
			responses.APIFormat = config.APIFormatResponses
			cfg.ModelConfig["model-fallback-1"] = responses
			chat := cfg.ModelConfig["model-fallback-2"]
			chat.APIFormat = config.APIFormatOpenAI
			cfg.ModelConfig["model-fallback-2"] = chat
			if failure == "rate_limit" {
				cfg.RateLimit.Providers = []config.RateLimitProviderConfig{{
					Type: "local-limiter",
					Rules: []config.RateLimitRule{{
						Name:            "cached-responses-candidate",
						Match:           config.RateLimitMatch{Model: "model-fallback-1"},
						RequestsPerUnit: 1, Unit: "minute",
					}},
				}}
				router.RateLimiter = buildRateLimitResolver(cfg)
			}

			request, envelope, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(
				llmprotocol.AnthropicMessagesV1, []byte(fallbackCachedAnthropicRequest),
			)
			if err != nil {
				t.Fatal(err)
			}
			ctx := testFallbackRequestContext("model-primary", []string{
				"model-primary", "model-fallback-1", "model-fallback-2",
			})
			ctx.SourceFormat = llmprotocol.AnthropicMessagesV1
			ctx.TargetFormat = llmprotocol.OpenAIChatV1
			ctx.SemanticRequest = &request
			ctx.ProtocolEnvelope = envelope
			ctx.Headers = map[string]string{}
			ctx.UpstreamStatusCode = http.StatusServiceUnavailable
			ctx.ProtocolDiagnostics = llmprotocol.Diagnostics{{
				Source: llmprotocol.AnthropicMessagesV1, Field: "baseline", Action: llmprotocol.DiagnosticGenerated,
			}}
			if failure == "rate_limit" {
				if response := router.applyRateLimit(ctx, "model-primary"); response != nil {
					t.Fatal("primary rate limit setup failed")
				}
				if response := router.applyRateLimit(&RequestContext{Headers: map[string]string{}}, "model-fallback-1"); response != nil {
					t.Fatal("candidate rate limit setup failed")
				}
			}

			var sawResponses, sawChat bool
			router.fallbackCaller = func(_ context.Context, model string, body []byte, _ map[string]string) ([]byte, int, error) {
				switch model {
				case "model-fallback-1":
					sawResponses = true
					if bytes.Contains(body, []byte("cache_control")) || !bytes.Contains(body, []byte(`"input"`)) {
						return nil, 0, fmt.Errorf("Responses fallback received wrong cache projection: %s", body)
					}
					return []byte(`{"error":{"message":"candidate unavailable","type":"server_error"}}`), http.StatusServiceUnavailable, nil
				case "model-fallback-2":
					sawChat = true
					if !bytes.Contains(body, []byte("cache_control")) || !bytes.Contains(body, []byte(`"messages"`)) {
						return nil, 0, fmt.Errorf("Chat fallback lost original cache boundary: %s", body)
					}
					return []byte(`{
						"id":"chatcmpl-cached-fallback","object":"chat.completion","model":"model-fallback-2",
						"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
						"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}
					}`), http.StatusOK, nil
				default:
					return nil, 0, fmt.Errorf("unexpected fallback candidate %q", model)
				}
			}

			response := router.handleUpstreamTransportError(
				[]byte(`{"error":{"message":"primary unavailable","type":"server_error"}}`), ctx,
			)
			if response == nil || response.GetImmediateResponse() == nil ||
				response.GetImmediateResponse().GetStatus().GetCode() != 200 {
				t.Fatalf("Chat fallback did not succeed: response=%+v", response)
			}
			if sawResponses != (failure == "upstream") || !sawChat || ctx.VSRSelectedModel != "model-fallback-2" {
				t.Fatalf("fallback sequence changed: sawResponses=%t sawChat=%t selected=%q", sawResponses, sawChat, ctx.VSRSelectedModel)
			}
			sawBaseline := false
			for _, diagnostic := range ctx.ProtocolDiagnostics {
				if diagnostic.Field == "cache_control" {
					t.Fatalf("failed Responses candidate leaked cache warning: %+v", ctx.ProtocolDiagnostics)
				}
				sawBaseline = sawBaseline || diagnostic.Field == "baseline"
			}
			if !sawBaseline {
				t.Fatalf("primary diagnostics were lost: %+v", ctx.ProtocolDiagnostics)
			}
		})
	}
}

func TestAnthropicCachedResponsesCandidateProgressesLikeSelection(t *testing.T) {
	router := routingTestRouter("chat")
	model := router.Config.ModelConfig["chat"]
	model.APIFormat = config.APIFormatResponses
	router.Config.ModelConfig["chat"] = model
	request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(
		llmprotocol.AnthropicMessagesV1,
		[]byte(`{"model":"chat","max_tokens":64,"messages":[{"role":"user","content":[{"type":"text","text":"hello","cache_control":{"type":"ephemeral"}}]}]}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	ref := config.ModelRef{Model: "chat"}
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, &request)
	selectionContext := &selection.SelectionContext{CandidateModels: []config.ModelRef{ref}}
	if err := router.candidateCapabilityMismatch(ref, &request, nil, nil, nil); err != nil {
		t.Fatalf("normal selection rejected Anthropic cached request: %v", err)
	}
	if reason := router.progressCandidateReason(ctx, selectionContext, &ref); reason != "" {
		t.Fatalf("progressive candidate rejected normal-selection-compatible Responses backend: %s", reason)
	}
}
