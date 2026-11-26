package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// streamChoicesProbe is one Chat client request for n choices, dispatched to a
// backend of the given format, optionally under a request_params cap on n.
type streamChoicesProbe struct {
	backend llmprotocol.WireFormat
	stream  bool
	n       *int64
	maxN    *int
}

func streamMultipleChoicesDispatch(t *testing.T, probe streamChoicesProbe) (*RequestContext, []byte, error) {
	t.Helper()
	router, logicalModel := routingTestRouterForFormat(probe.backend)
	request := testNeutralRequest(logicalModel, "hello")
	request.Stream = probe.stream
	request.CandidateCount = probe.n
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.ExpectStreamingResponse = probe.stream
	if probe.maxN != nil {
		payload, err := config.NewStructuredPayload(config.RequestParamsPluginConfig{MaxN: probe.maxN})
		require.NoError(t, err)
		ctx.VSRSelectedDecision = &config.Decision{
			Name:    "capped-candidates",
			Plugins: []config.DecisionPlugin{{Type: "request_params", Configuration: payload}},
		}
	}
	response, err := router.handleEntrypointModelRouting(request, logicalModel, "", entropy.ReasoningDecision{}, logicalModel, ctx)
	if err != nil {
		require.Nil(t, response, "a rejected dispatch must not produce an upstream body")
		return ctx, nil, err
	}
	return ctx, response.GetRequestBody().GetResponse().GetBodyMutation().GetBody(), nil
}

// The Chat stream decoder accepts only choice index 0, so n>1 with stream:true
// must be rejected before the request reaches the backend instead of failing
// at the backend's end of stream.
func TestChatDispatchRejectsStreamingMultipleChoices(t *testing.T) {
	ctx, body, err := streamMultipleChoicesDispatch(t, streamChoicesProbe{
		backend: llmprotocol.OpenAIChatV1, stream: true, n: llmprotocol.Int64(2),
	})
	require.Error(t, err, "n=2 with stream:true was dispatched: %s", body)
	var protocolError *llmprotocol.ProtocolError
	require.ErrorAs(t, err, &protocolError)
	require.Equal(t, llmprotocol.ErrorUnsupportedFeature, protocolError.Category)
	require.Equal(t, "stream_multiple_choices", protocolError.Code)

	router := &OpenAIRouter{}
	response, converted := router.processBodyRoutingError(err, ctx)
	require.True(t, converted)
	require.EqualValues(t, 400, response.GetImmediateResponse().GetStatus().GetCode())
	client := router.encodeImmediateResponseForClient(response, ctx).GetImmediateResponse().GetBody()
	require.Contains(t, string(client), `"code":"stream_multiple_choices"`)
	require.Contains(t, string(client), `"type":"invalid_request_error"`)
}

// Every streamed candidate count above one is refused, not only n=2, and the
// refusal is recorded for the client's protocol.
func TestChatDispatchRejectsEveryStreamedCandidateCountAboveOne(t *testing.T) {
	for _, n := range []int64{2, 3, 32} {
		t.Run(fmt.Sprintf("n%d", n), func(t *testing.T) {
			ctx, body, err := streamMultipleChoicesDispatch(t, streamChoicesProbe{
				backend: llmprotocol.OpenAIChatV1, stream: true, n: llmprotocol.Int64(n),
			})
			require.Error(t, err, "n=%d with stream:true was dispatched: %s", n, body)
			var protocolError *llmprotocol.ProtocolError
			require.ErrorAs(t, err, &protocolError)
			require.Equal(t, llmprotocol.ErrorUnsupportedFeature, protocolError.Category)
			require.Equal(t, "stream_multiple_choices", protocolError.Code)
			require.Same(t, protocolError, ctx.ImmediateProtocolError)
		})
	}
}

// Native fallback qualifies each candidate with the same dispatch check, so a
// streamed n>1 request never becomes a fallback call to a Chat candidate.
func TestNativeFallbackSkipsStreamingMultipleChoices(t *testing.T) {
	for _, tc := range []struct {
		name     string
		n        int64
		wantCall bool
	}{{"n2", 2, false}, {"n1", 1, true}} {
		t.Run(tc.name, func(t *testing.T) {
			session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), "model-primary", "model-fallback-1")
			session.ctx.SemanticRequest.Stream = true
			session.ctx.SemanticRequest.CandidateCount = llmprotocol.Int64(tc.n)
			session.ctx.ExpectStreamingResponse = true
			step, err := session.Fallback().Next(context.Background(), failedWith(http.StatusServiceUnavailable, "unavailable"))
			require.NoError(t, err)
			if !tc.wantCall {
				require.Nil(t, step.Call, "fallback body: %s", fallbackCallBody(step.Call))
				return
			}
			require.NotNil(t, step.Call)
			require.Equal(t, "model-fallback-1", step.Call.Route)
			require.Contains(t, string(step.Call.Request.Body), `"stream":true`)
		})
	}
}

func fallbackCallBody(call *routing.Call) string {
	if call == nil {
		return ""
	}
	return string(call.Request.Body)
}

// Requests whose reply the Chat decoders can read still reach the backend: one
// streamed choice, several buffered choices, and a streamed request whose
// request_params cap lowers n to 1 before dispatch.
func TestChatDispatchKeepsReadableCandidateRequests(t *testing.T) {
	for _, tc := range []struct {
		name  string
		probe streamChoicesProbe
		wantN int64
	}{
		{"one streamed choice", streamChoicesProbe{stream: true, n: llmprotocol.Int64(1)}, 1},
		{"several buffered choices", streamChoicesProbe{stream: false, n: llmprotocol.Int64(2)}, 2},
		{"streamed choices capped to one", streamChoicesProbe{stream: true, n: llmprotocol.Int64(2), maxN: extprocIntPtr(1)}, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tc.probe.backend = llmprotocol.OpenAIChatV1
			_, body, err := streamMultipleChoicesDispatch(t, tc.probe)
			require.NoError(t, err)
			var sent struct {
				N      *int64 `json:"n"`
				Stream bool   `json:"stream"`
			}
			require.NoError(t, json.Unmarshal(body, &sent), "backend body: %s", body)
			require.NotNil(t, sent.N, "backend body lost n: %s", body)
			require.Equal(t, tc.wantN, *sent.N, "backend body: %s", body)
			require.Equal(t, tc.probe.stream, sent.Stream, "backend body: %s", body)
		})
	}
}

// A backend format that cannot carry n keeps the capability error it reported
// before the stream check existed.
func TestStreamingMultipleChoicesToMessagesKeepsTheCapabilityError(t *testing.T) {
	_, body, err := streamMultipleChoicesDispatch(t, streamChoicesProbe{
		backend: llmprotocol.AnthropicMessagesV1, stream: true, n: llmprotocol.Int64(2),
	})
	require.Error(t, err, "n=2 reached a Messages backend: %s", body)
	var protocolError *llmprotocol.ProtocolError
	require.ErrorAs(t, err, &protocolError)
	require.Equal(t, "unsupported_capability", protocolError.Code)
	require.Contains(t, protocolError.Message, "multiple_candidates")
}
