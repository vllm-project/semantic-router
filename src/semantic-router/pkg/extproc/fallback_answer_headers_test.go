package extproc

import (
	"context"
	"net/http"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// A response served by cross-model fallback carries the final routing facts
// of any routed response in both gateway modes: Envoy mode's response-phase
// fallback answers with an immediate response, the native gateway passes the
// candidate's response through the response phases. In both tests the
// primary answers 503 and the candidate 200.

const fallbackCandidateAnswer = `{"id":"chatcmpl-fallback","object":"chat.completion","created":1700000000,` +
	`"model":"model-fallback-1","choices":[{"index":0,"message":{"role":"assistant","content":"From the candidate."},` +
	`"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}`

// withRoutingFacts gives a fallback test context what routing decided before
// the primary was called.
func withRoutingFacts(ctx *RequestContext) *RequestContext {
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: "support"})
	ctx.VSRSelectionMethod = "static"
	ctx.RoutingLatency = 1500 * time.Microsecond
	return ctx
}

func upstreamStatus(status string) *ext_proc.ProcessingRequest_ResponseHeaders {
	return &ext_proc.ProcessingRequest_ResponseHeaders{ResponseHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: []*core.HeaderValue{{Key: ":status", Value: status}}},
	}}
}

func requireFallbackAnswerHeaders(t *testing.T, set []*core.HeaderValueOption) {
	t.Helper()
	got := map[string]string{}
	for _, option := range set {
		got[option.GetHeader().GetKey()] = string(option.GetHeader().GetRawValue())
	}
	for name, want := range map[string]string{
		headers.VSRSelectedRecipe:    "support",
		headers.VSRSelectedDecision:  "chat_fallback_decision",
		headers.VSRSelectedAlgorithm: "static",
		headers.VSRRoutingLatencyMs:  "1.500",
		headers.VSRSelectedModel:     "model-fallback-1",
		headers.VSRFallbackAttempts:  "2",
		headers.VSRResponsePath:      headers.ResponsePathFallback,
	} {
		require.Equal(t, want, got[name], "%s in %v", name, got)
	}
}

func TestResponsePhaseFallbackAnswerCarriesTheRoutingFacts(t *testing.T) {
	router, _ := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
	ctx := withRoutingFacts(testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"}))
	router.fallbackCaller = func(context.Context, string, []byte, map[string]string) ([]byte, int, error) {
		return []byte(fallbackCandidateAnswer), http.StatusOK, nil
	}

	response, err := router.handleResponseHeaders(upstreamStatus("503"), ctx)
	require.NoError(t, err)
	immediate := response.GetImmediateResponse()
	require.NotNil(t, immediate, "the candidate's answer replaces the primary's 503")
	requireFallbackAnswerHeaders(t, immediate.GetHeaders().GetSetHeaders())
}

func TestNativeFallbackAnswerCarriesTheRoutingFacts(t *testing.T) {
	session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), "model-primary", "model-fallback-1")
	ctx := withRoutingFacts(session.ctx)
	step, err := session.Fallback().Next(context.Background(), failedWith(http.StatusServiceUnavailable, "unavailable"))
	require.NoError(t, err)
	require.NotNil(t, step.Call)
	session.fallback.succeeded(http.StatusOK)

	response, err := session.router.handleResponseHeaders(upstreamStatus("200"), ctx)
	require.NoError(t, err)
	mutation := response.GetResponseHeaders().GetResponse().GetHeaderMutation()
	require.NotNil(t, mutation, "the candidate's response headers")
	requireFallbackAnswerHeaders(t, mutation.GetSetHeaders())
}
