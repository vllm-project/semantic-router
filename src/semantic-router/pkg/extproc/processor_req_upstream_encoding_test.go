package extproc

import (
	"context"
	"net/http"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

func TestAcceptsGzip(t *testing.T) {
	accepted := []string{
		"gzip", "GZIP", "x-gzip", "gzip, deflate, br", "deflate, gzip;q=0.5", "gzip;Q=1",
		" gzip ; q=0.8 ", "gzip;q=1.000", "gzip;q=0.001", "*", "br;q=1, *;q=0.5", "*;q=0, gzip",
	}
	refused := []string{
		"", "identity", "br, zstd", "deflate", "gzip;q=0", "gzip;q=0.000", "gzip;q=0, *",
		"*;q=0", "x-gzip;q=0, *", "gzip;q=abc", "gzip;q=", "gzip;q=1.5", "gzip;q=1.001",
		"gzip;q=0.1234", "gzip;q=-1", "gzip;q=.5", "gzip;q=1e0",
	}
	for _, value := range accepted {
		require.True(t, acceptsGzip(value), "Accept-Encoding %q should accept gzip", value)
	}
	for _, value := range refused {
		require.False(t, acceptsGzip(value), "Accept-Encoding %q should refuse gzip", value)
	}
}

// The provider is asked for gzip only when the router reads the whole reply and
// its consumer accepts gzip: a buffered reply to a client that asked for gzip,
// or a Looper hop, whose reply the router itself consumes. Everything else keeps
// the request-header phase's identity default (an empty value on the wire).
func TestProviderDispatchNegotiatesUpstreamEncoding(t *testing.T) {
	tests := []struct {
		name      string
		client    []string
		streaming bool
		hop       bool
		want      string
	}{
		{name: "client accepts gzip", client: []string{"gzip, deflate, br"}, want: "gzip"},
		{name: "client accepts any coding", client: []string{"*"}, want: "gzip"},
		{name: "client states no preference", want: ""},
		{name: "client asks for identity", client: []string{"identity"}, want: ""},
		{name: "client refuses gzip", client: []string{"gzip;q=0, identity"}, want: ""},
		{name: "client accepts no coding", client: []string{""}, want: ""},
		{name: "streamed reply", client: []string{"gzip"}, streaming: true, want: ""},
		{name: "looper hop", hop: true, want: "gzip"},
		{name: "streamed looper hop", hop: true, streaming: true, want: ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			router := routingTestRouter("worker")
			request := testNeutralRequest("worker", "Summarize this note.")
			request.Stream = tc.streaming
			ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
			ctx.ExpectStreamingResponse = tc.streaming
			ctx.LooperRequest = tc.hop
			client := routing.Header{{Name: "content-type", Value: "application/json"}}
			for _, value := range tc.client {
				ctx.Headers["accept-encoding"] = value
				client = append(client, routing.HeaderField{Name: "accept-encoding", Value: value})
			}

			var response *ext_proc.ProcessingResponse
			var err error
			if tc.hop {
				response, err = router.handleLooperInternalRequest("worker", ctx)
			} else {
				response, err = router.handleSpecifiedModelRouting(request, "worker", "", ctx)
			}
			require.NoError(t, err)
			require.NotNil(t, response.GetRequestBody(), "expected a provider dispatch, got %v", response)
			require.Equal(t, tc.want, providerAcceptEncoding(t, client, response))
		})
	}
}

func TestProviderDispatchNegotiatesUpstreamEncodingOnEveryRoutedPath(t *testing.T) {
	for _, flow := range []string{"same_model", "specified_model"} {
		t.Run(flow, func(t *testing.T) {
			router := routingTestRouter("worker")
			request := testNeutralRequest("worker", "Summarize this note.")
			ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
			ctx.Headers["accept-encoding"] = "gzip"
			var response *ext_proc.ProcessingResponse
			var err error
			if flow == "same_model" {
				response, err = router.handleEntrypointModelRouting(request, "worker", "", entropy.ReasoningDecision{}, "worker", ctx)
			} else {
				response, err = router.handleSpecifiedModelRouting(request, "worker", "", ctx)
			}
			require.NoError(t, err)
			client := routing.Header{{Name: "accept-encoding", Value: "gzip"}}
			require.Equal(t, "gzip", providerAcceptEncoding(t, client, response))
		})
	}
}

// A native fallback candidate is dispatched as the primary would have been,
// so it negotiates the reply encoding the same way.
func TestNativeFallbackCandidateNegotiatesUpstreamEncoding(t *testing.T) {
	for _, tc := range []struct{ name, client, want string }{
		{name: "client accepts gzip", client: "gzip, deflate", want: "gzip"},
		{name: "client asks for identity", client: "identity", want: ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), "model-primary", "model-fallback-1")
			session.ctx.Headers = map[string]string{"accept-encoding": tc.client}
			// The request-header phase left identity, which Envoy applies as an
			// empty value.
			session.clientHeader = append(session.clientHeader, routing.HeaderField{Name: "accept-encoding", Value: ""})

			step, err := session.Fallback().Next(context.Background(), failedWith(http.StatusServiceUnavailable, "unavailable"))
			require.NoError(t, err)
			require.NotNil(t, step.Call)
			require.Equal(t, tc.want, step.Call.Request.Header.Get("accept-encoding"))
		})
	}
}

// In FULL_DUPLEX_STREAMED mode the body stage's mutations ride on the held
// header reply after the header stage's own, so the negotiated gzip wins.
func TestFullDuplexHeaderReplyNegotiatesUpstreamEncoding(t *testing.T) {
	router := fullDuplexRoutingRouter()
	ctx := &RequestContext{Headers: make(map[string]string)}
	stream := NewMockStream(nil)
	accept := &core.HeaderValue{Key: "accept-encoding", RawValue: []byte("gzip, deflate")}

	runProcessRequests(t, router, ctx, stream, fullDuplexHeadersRequest(false, accept), bodyRequest(fullDuplexTestBody, true))
	require.Len(t, stream.Responses, 2)
	header := routing.Header{{Name: "accept-encoding", Value: "gzip, deflate"}}
	effect, err := routingEffect(stream.Responses[0])
	require.NoError(t, err)
	require.NoError(t, routing.ApplyHeaderMutation(&header, effect.Header, false, routing.DefaultLimits))
	require.Equal(t, "gzip", header.Get("accept-encoding"))
}

// Provider extra headers are operator policy and are applied after the
// negotiated encoding, so an explicit operator choice still wins.
func TestProviderExtraHeadersOverrideNegotiatedEncoding(t *testing.T) {
	router := routingTestRouter("worker")
	profile := router.Config.ProviderProfiles["provider"]
	profile.ExtraHeaders = map[string]string{"Accept-Encoding": "identity"}
	router.Config.ProviderProfiles["provider"] = profile
	request := testNeutralRequest("worker", "Summarize this note.")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.Headers["accept-encoding"] = "gzip"

	response, err := router.handleSpecifiedModelRouting(request, "worker", "", ctx)
	require.NoError(t, err)
	client := routing.Header{{Name: "accept-encoding", Value: "gzip"}}
	require.Equal(t, "identity", providerAcceptEncoding(t, client, response))
}

// An external gateway owns the backend and may read the reply itself, so the
// router does not change the encoding that gateway negotiates.
func TestExternalGatewayDispatchKeepsIdentityEncoding(t *testing.T) {
	const model = "openai/gpt-oss-20b"
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			model: {APIFormat: config.APIFormatOpenAI},
		}},
	}}
	request := testNeutralRequest(model, "continue the conversation")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.Headers["accept-encoding"] = "gzip"

	response, err := router.handleModelRouting(request, model, "", entropy.ReasoningDecision{}, "", ctx)
	require.NoError(t, err)
	require.NotNil(t, response.GetRequestBody(), "expected an external gateway dispatch, got %v", response)
	client := routing.Header{{Name: "accept-encoding", Value: "gzip"}}
	require.Empty(t, providerAcceptEncoding(t, client, response))
}

// providerAcceptEncoding applies the request-header phase and the given
// request-body replies to the client's headers, as Envoy and the routing
// codec do (both apply raw_value only), and returns the Accept-Encoding the
// provider receives.
func providerAcceptEncoding(t *testing.T, client routing.Header, bodyReplies ...*ext_proc.ProcessingResponse) string {
	t.Helper()
	header := client.Clone()
	replies := append([]*ext_proc.ProcessingResponse{
		newContinueRequestHeadersResponse(buildIdentityEncodingRequestMutation()),
	}, bodyReplies...)
	for _, reply := range replies {
		effect, err := routingEffect(reply)
		require.NoError(t, err)
		require.NoError(t, routing.ApplyHeaderMutation(&header, effect.Header, false, routing.DefaultLimits))
	}
	return header.Get("accept-encoding")
}
