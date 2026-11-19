package extproc

import (
	"fmt"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// Typed OpenAI clients send "parameters": null for a function that takes no
// arguments. The fixtures take the model and the tool's parameters member.
var nullToolParameterBodies = map[llmprotocol.WireFormat]string{
	llmprotocol.OpenAIChatV1:      `{"model":%q,"messages":[{"role":"user","content":"What time is it?"}],"tools":[{"type":"function","function":{"name":"get_time","strict":false%s}}]}`,
	llmprotocol.OpenAIResponsesV1: `{"model":%q,"input":"What time is it?","tools":[{"type":"function","name":"get_time","strict":false%s}]}`,
}

func TestNullToolParametersReachEveryBackendLikeOmittedParameters(t *testing.T) {
	for client := range nullToolParameterBodies {
		for _, backend := range extProcMatrixFormats {
			t.Run(string(client)+"_client_"+string(backend)+"_backend", func(t *testing.T) {
				omitted := dispatchNullToolParametersRequest(t, client, backend, "")
				null := dispatchNullToolParametersRequest(t, client, backend, `,"parameters":null`)
				require.Equal(t, string(omitted), string(null))

				decoded, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(backend, null)
				require.NoError(t, err, string(null))
				require.Len(t, decoded.Tools, 1)
				require.Equal(t, "get_time", decoded.Tools[0].Name)
				require.JSONEq(t, `{"type":"object"}`, string(decoded.Tools[0].InputSchema))
			})
		}
	}
}

func dispatchNullToolParametersRequest(t *testing.T, client, backend llmprotocol.WireFormat, parameters string) []byte {
	t.Helper()
	router, model := routingTestRouterForFormat(backend)
	router.Cache = cache.NewInMemoryCache(cache.InMemoryCacheOptions{Enabled: false})
	ctx := &RequestContext{Headers: map[string]string{}, RequestID: "null-tool-parameters", SourceFormat: client}
	body := fmt.Sprintf(nullToolParameterBodies[client], model, parameters)
	response, err := router.HandleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(body)},
	}, ctx)
	require.NoError(t, err)
	if immediate := router.encodeImmediateResponseForClient(response, ctx).GetImmediateResponse(); immediate != nil {
		t.Fatalf("request was refused: %s", immediate.GetBody())
	}
	dispatched := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	require.NotEmpty(t, dispatched)
	return dispatched
}

// A same-format request that nothing else changes is byte-replay eligible, but
// its bytes still carry the null, so the backend receives the encoded schema.
func TestNullToolParametersAreNotReplayedToASameFormatBackend(t *testing.T) {
	for format, fixture := range nullToolParameterBodies {
		t.Run(string(format), func(t *testing.T) {
			router := &OpenAIRouter{}
			ctx := &RequestContext{SourceFormat: format, TargetFormat: format, RequestID: "null-tool-parameters", TraceContext: t.Context()}
			request, immediate := router.prepareProtocolRequest([]byte(fmt.Sprintf(fixture, "public-model", `,"parameters":null`)), ctx)
			require.Nil(t, immediate)
			require.NotNil(t, request)
			dispatched, err := router.encodeDispatchRequest(ctx)
			require.NoError(t, err)
			require.NotContains(t, string(dispatched), `"parameters":null`)
			require.Contains(t, string(dispatched), `"parameters":{"type":"object"}`)
		})
	}
}
