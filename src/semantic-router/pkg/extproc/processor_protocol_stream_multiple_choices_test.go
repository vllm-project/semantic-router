package extproc

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
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
