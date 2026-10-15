package extproc

import (
	"context"
	"fmt"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestNativeFallbackValidatesDynamoCandidate(t *testing.T) {
	for _, backend := range []string{"vllm", "dynamo"} {
		for _, source := range []string{"nvext", "header", "profile"} {
			t.Run(backend+"/"+source, func(t *testing.T) {
				session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), "model-primary", "model-fallback-1")
				session.router.Config.VLLMEndpoints[0].Type = "dynamo"
				session.router.Config.VLLMEndpoints[1].Type = backend
				switch source {
				case "nvext":
					session.ctx.ProtocolEnvelope = llmprotocol.Envelope{Format: llmprotocol.OpenAIChatV1, Dynamo: &llmprotocol.DynamoEnvelope{RequestNVExt: &llmprotocol.DynamoRequestNVExt{CacheSalt: "private"}}}
				case "header":
					// The actual outbound header can exist only in the native transport snapshot.
					session.clientHeader.Set(headers.DynamoDPRank, "1")
				case "profile":
					profile := session.router.Config.ProviderProfiles["prof-openai"]
					profile.ExtraHeaders = map[string]string{headers.DynamoDPRank: "1"}
					session.router.Config.ProviderProfiles["prof-openai"] = profile
				}
				step, err := session.Fallback().Next(context.Background(), failedWith(http.StatusServiceUnavailable, "unavailable"))
				require.NoError(t, err)
				if backend == "vllm" {
					require.Nil(t, step.Call, "must not dispatch Dynamo state to a non-Dynamo fallback")
				} else {
					require.NotNil(t, step.Call)
					if source == "nvext" {
						require.Contains(t, string(step.Call.Request.Body), `"cache_salt":"private"`)
					}
					if source != "nvext" {
						require.Equal(t, "1", step.Call.Request.Header.Get(headers.DynamoDPRank))
					}
				}
			})
		}
	}
}

func TestResponseFallbackRejectsDynamoRequestBeforeSending(t *testing.T) {
	router, cfg := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
	cfg.VLLMEndpoints[0].Type = "dynamo"
	cfg.VLLMEndpoints[1].Type = "vllm"
	ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
	ctx.UpstreamStatusCode = http.StatusServiceUnavailable
	ctx.ProtocolEnvelope = llmprotocol.Envelope{Format: llmprotocol.OpenAIChatV1, Dynamo: &llmprotocol.DynamoEnvelope{RequestNVExt: &llmprotocol.DynamoRequestNVExt{CacheSalt: "private"}}}
	called := false
	router.fallbackCaller = func(context.Context, string, []byte, map[string]string) ([]byte, int, error) {
		called = true
		return nil, http.StatusOK, nil
	}
	require.Nil(t, router.maybeExecuteFallback([]byte("unavailable"), ctx))
	require.False(t, called, "request must be rejected before transport")
}

func TestResponseFallbackValidatesDynamoResponse(t *testing.T) {
	for _, streamed := range []bool{false, true} {
		for _, backend := range []string{"vllm", "dynamo"} {
			t.Run(fmt.Sprintf("stream=%t/%s", streamed, backend), func(t *testing.T) {
				router, cfg := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
				cfg.VLLMEndpoints[0].Type = "dynamo"
				cfg.VLLMEndpoints[1].Type = backend
				ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
				ctx.UpstreamStatusCode = http.StatusServiceUnavailable
				ctx.IsStreamingResponse = streamed
				ctx.SemanticRequest.Stream = streamed
				// Stale permission from the primary must not authorize the fallback.
				ctx.AllowDynamoExtensions = backend != "dynamo"
				ctx.UpstreamBackendName = "backend-primary"
				called := false
				router.fallbackCaller = func(context.Context, string, []byte, map[string]string) ([]byte, int, error) {
					called = true
					return []byte(`{"id":"fallback","object":"chat.completion","model":"model-fallback-1","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"nvext":{"token_ids":[1]}}`), http.StatusOK, nil
				}
				response := router.maybeExecuteFallback([]byte("unavailable"), ctx)
				require.True(t, called)
				if backend == "vllm" {
					require.Nil(t, response, "forbidden metadata must not become an immediate response or SSE")
					require.NotEqual(t, "succeeded", ctx.FallbackRecord.FinalStatus)
					require.Nil(t, ctx.SemanticResponse)
					require.Contains(t, ctx.FallbackRecord.Attempts[1].Error.Error(), "unexpected_dynamo_nvext_backend")
				} else {
					require.NotNil(t, response)
					require.Equal(t, "succeeded", ctx.FallbackRecord.FinalStatus)
					require.Contains(t, string(response.GetImmediateResponse().Body), "hello")
					if streamed {
						require.Contains(t, string(response.GetImmediateResponse().Body), "data:")
					}
				}
			})
		}
	}
}
