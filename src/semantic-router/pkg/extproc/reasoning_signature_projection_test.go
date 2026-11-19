package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// thinkingHistory is a Messages conversation whose second message holds the
// thinking an Anthropic backend wrote, with the given signature.
const thinkingHistory = `{"model":"vllm-sr/auto","max_tokens":64,"messages":[
	{"role":"user","content":"What is 2+2?"},
	{"role":"assistant","content":[{"type":"thinking","thinking":"simple arithmetic","signature":%s},{"type":"text","text":"4"}]},
	{"role":"user","content":"And 3+3?"}]}`

const signatureApproximated = `protocol;action=approximated;reason="the selected backend cannot accept a reasoning signature; the reasoning text is kept";field="reasoning.signature"`

func thinkingHistoryRouter(t *testing.T, backend llmprotocol.WireFormat, lossy string) *OpenAIRouter {
	t.Helper()
	apiFormat, provider := config.APIFormatOpenAI, "openai"
	switch backend {
	case llmprotocol.OpenAIResponsesV1:
		apiFormat = config.APIFormatResponses
	case llmprotocol.AnthropicMessagesV1:
		apiFormat, provider = config.APIFormatAnthropic, "anthropic"
	}
	cfg, err := config.ParseYAMLBytes([]byte(fmt.Sprintf(`
version: v0.5
providers:
  defaults:
    model: worker
  models:
    - name: worker
      provider_model_id: worker
      api_format: %s
      backend_refs: [{name: b, provider: %s, endpoint: "https://api.example.com/v1"}]
recipes:
  - name: default
    decisions:
      - name: history
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: worker}]
global:
  router:
    allow_lossy_translation: %s
`, apiFormat, provider, lossy)))
	require.NoError(t, err)
	router, err := buildOpenAIRouterFromConfig(cfg)
	require.NoError(t, err)
	t.Cleanup(func() { _ = router.Close() })
	router.CredentialResolver = newTestCredentialResolver(cfg)
	return router
}

// sendThinkingHistory runs the conversation through the request phases and
// returns the request-body reply.
func sendThinkingHistory(t *testing.T, router *OpenAIRouter, signature string) (*ext_proc.ProcessingResponse, *RequestContext) {
	t.Helper()
	ctx := &RequestContext{Headers: map[string]string{}}
	require.NoError(t, router.processRequestHeaders(NewMockStream(nil), newRequestHeaders("POST", "/v1/messages"), ctx))
	stream := NewMockStream(nil)
	require.NoError(t, router.processRequestBody(stream, &ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(fmt.Sprintf(thinkingHistory, strconv.Quote(signature))), EndOfStream: true},
	}, ctx))
	require.Len(t, stream.Responses, 1)
	return stream.Responses[0], ctx
}

// answerFromBackend runs the backend's buffered reply through the response
// phases and returns x-vsr-warnings and the body the client receives.
func answerFromBackend(t *testing.T, router *OpenAIRouter, ctx *RequestContext, reply []byte) (string, []byte) {
	t.Helper()
	stream := NewMockStream(nil)
	require.NoError(t, router.processResponseHeaders(stream, &ext_proc.ProcessingRequest{}, &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":status", Value: "200"}, {Key: "content-type", Value: "application/json"},
		}}},
	}, ctx))
	require.NoError(t, router.processResponseBody(stream, &ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: reply, EndOfStream: true},
	}, ctx))
	require.Len(t, stream.Responses, 2)
	require.Nil(t, stream.Responses[1].GetImmediateResponse(), "the backend's answer was refused")
	set := headerValuesByName(stream.Responses[0].GetResponseHeaders().GetResponse().GetHeaderMutation().GetSetHeaders())
	mutation := stream.Responses[1].GetResponseBody().GetResponse().GetBodyMutation()
	if mutation == nil {
		return set[headers.VSRWarnings], reply // A same-format reply reaches the client as the backend sent it.
	}
	return set[headers.VSRWarnings], mutation.GetBody()
}

// A session that moves from an Anthropic backend sends back the thinking that
// backend signed. Once the operator accepts the loss, a Chat or Responses
// backend receives the thinking text without the signature, the client gets
// the answer, and x-vsr-warnings reports the loss. An Anthropic backend keeps
// the signature, and an empty signature needs no opt-in.
func TestSignedThinkingHistoryReachesEveryBackendFormat(t *testing.T) {
	for _, test := range []struct {
		name      string
		backend   llmprotocol.WireFormat
		lossy     string
		signature string
		keeps     bool
		warning   string
	}{
		{"chat", llmprotocol.OpenAIChatV1, "[reasoning.signature]", "EqQBCkYIBxgCKkBexample", false, signatureApproximated},
		{"responses", llmprotocol.OpenAIResponsesV1, "[reasoning.signature]", "EqQBCkYIBxgCKkBexample", false, signatureApproximated},
		{"messages", llmprotocol.AnthropicMessagesV1, "[reasoning.signature]", "EqQBCkYIBxgCKkBexample", true, ""},
		{"chat, empty signature", llmprotocol.OpenAIChatV1, "[]", "", false, ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			router := thinkingHistoryRouter(t, test.backend, test.lossy)
			reply, ctx := sendThinkingHistory(t, router, test.signature)
			require.Nil(t, reply.GetImmediateResponse(), "the conversation was refused")
			sent := reply.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
			require.Contains(t, string(sent), "simple arithmetic", "the thinking text was lost")
			require.Contains(t, string(sent), "And 3+3?")
			require.Equal(t, test.keeps, strings.Contains(string(sent), `"signature":"EqQBCkYIBxgCKkBexample"`), "backend body: %s", sent)
			if !test.keeps {
				require.NotContains(t, string(sent), "signature")
			}
			require.Equal(t, test.signature, ctx.SemanticRequest.Messages[1].Content[0].Signature, "the client's request was changed")
			warnings, body := answerFromBackend(t, router, ctx, extProcResponseFixture(test.backend))
			require.Equal(t, test.warning, warnings)
			answer, _, _, err := protocolcodec.NewBuiltinEngine().DecodeResponse(llmprotocol.AnthropicMessagesV1, body)
			require.NoError(t, err, "client body: %s", body)
			require.Equal(t, "hello", answer.Output[0].Content[0].Text)
		})
	}
}

// A fallback candidate is qualified and encoded through the same projection,
// so when an Anthropic primary fails, a Chat candidate receives the thinking
// without the signature once the loss is accepted.
func TestSignedThinkingHistoryReachesAChatFallbackCandidate(t *testing.T) {
	router, cfg := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
	cfg.AllowLossyTranslation = []string{llmprotocol.LossyFieldReasoningSignature}
	request, envelope, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1,
		[]byte(fmt.Sprintf(thinkingHistory, strconv.Quote("EqQBCkYIBxgCKkBexample"))))
	require.NoError(t, err)
	ctx := testFallbackRequestContext("model-fallback-2", []string{"model-fallback-2", "model-fallback-1"})
	ctx.SourceFormat, ctx.TargetFormat = llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1
	ctx.SemanticRequest, ctx.ProtocolEnvelope = &request, envelope
	ctx.Headers = map[string]string{}
	ctx.UpstreamStatusCode = http.StatusServiceUnavailable
	var sent []byte
	router.fallbackCaller = func(_ context.Context, _ string, body []byte, _ map[string]string) ([]byte, int, error) {
		sent = body
		return extProcResponseFixture(llmprotocol.OpenAIChatV1), http.StatusOK, nil
	}
	response := router.handleUpstreamTransportError([]byte(`{"type":"error","error":{"type":"overloaded_error","message":"primary unavailable"}}`), ctx)
	require.EqualValues(t, http.StatusOK, response.GetImmediateResponse().GetStatus().GetCode())
	require.Equal(t, "model-fallback-1", ctx.VSRSelectedModel)
	require.Contains(t, string(sent), "simple arithmetic")
	require.NotContains(t, string(sent), "signature")
	require.Contains(t, responseWarnings(ctx), signatureApproximated)
}

// Without the opt-in the conversation is refused before routing with the
// client's own field, not the Router's wire format or capability names.
func TestSignedThinkingHistoryRefusalNamesTheClientField(t *testing.T) {
	for _, backend := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		t.Run(string(backend), func(t *testing.T) {
			reply, _ := sendThinkingHistory(t, thinkingHistoryRouter(t, backend, "[]"), "EqQBCkYIBxgCKkBexample")
			immediate := reply.GetImmediateResponse()
			require.NotNil(t, immediate, "the conversation was dispatched")
			require.EqualValues(t, 400, immediate.GetStatus().GetCode())
			var refusal struct {
				Type  string `json:"type"`
				Error struct {
					Type    string `json:"type"`
					Message string `json:"message"`
				} `json:"error"`
			}
			require.NoError(t, json.Unmarshal(immediate.GetBody(), &refusal), "body: %s", immediate.GetBody())
			require.Equal(t, "error", refusal.Type)
			require.Equal(t, "invalid_request_error", refusal.Error.Type)
			require.True(t, strings.HasPrefix(refusal.Error.Message,
				"messages[1].content[0].signature: the selected model cannot accept signed thinking"), refusal.Error.Message)
			require.Contains(t, refusal.Error.Message, "reasoning.signature in global.router.allow_lossy_translation")
			for _, internal := range []string{string(backend), "reasoning_signature"} {
				require.NotContains(t, refusal.Error.Message, internal)
			}
		})
	}
}
