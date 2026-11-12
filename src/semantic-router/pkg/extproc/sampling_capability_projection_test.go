package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// samplingCapabilityConfig serves the chat decision from two vLLM models:
// draft-model runs speculative decoding, which rejects min_p and logit_bias,
// and plain-model accepts both. The blocked decision removes logit_bias for
// every model it routes to.
const samplingCapabilityConfig = `
version: v0.5
providers:
  models:
    - name: draft-model
      provider_model_id: draft-model
      api_format: chat_completions
      unsupported_capabilities: [sampling_min_p, sampling_logit_bias]
      backend_refs:
        - name: draft-backend
          endpoint: 127.0.0.1:18001
          protocol: http
          provider: vllm
    - name: plain-model
      provider_model_id: plain-model
      api_format: chat_completions
      backend_refs:
        - name: plain-backend
          endpoint: 127.0.0.1:18002
          protocol: http
          provider: vllm
recipes:
  - name: default
    decisions:
      - name: chat
        priority: 10
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: draft-model
          - model: plain-model
      - name: blocked
        priority: 5
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: plain-model
        plugins:
          - type: request_params
            configuration:
              blocked_params: [logit_bias]
`

// speculativeDecodingRejection is what a vLLM backend that runs speculative
// decoding answers a request that carries min_p or logit_bias.
const speculativeDecodingRejection = `{"error":{"message":"The min_p and logit_bias sampling parameters are not yet supported with speculative decoding.","type":"BadRequestError","param":null,"code":400}}`

const samplingCandidateAnswer = `{"id":"chatcmpl-color","object":"chat.completion","created":1700000000,` +
	`"model":"candidate","choices":[{"index":0,"message":{"role":"assistant","content":"Teal is a color."},` +
	`"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":4,"total_tokens":9}}`

func samplingCapabilityRouter(t *testing.T) *OpenAIRouter {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(samplingCapabilityConfig))
	require.NoError(t, err)
	policy := fallback.DefaultEnabledPolicy()
	return &OpenAIRouter{
		Config:               cfg,
		CredentialResolver:   newTestCredentialResolver(cfg),
		FallbackOrchestrator: fallback.NewOrchestrator(policy, fallback.NewBackendCircuitBreaker(policy.CircuitBreaker)),
	}
}

// samplingRequestContext decodes a Chat request that carries min_p and
// logit_bias, as the body phase does, under the named decision.
func samplingRequestContext(t *testing.T, router *OpenAIRouter, decisionName string, stream bool) *RequestContext {
	t.Helper()
	body := fmt.Sprintf(`{"model":"vllm-sr/auto","stream":%t,"min_p":0.05,"logit_bias":{"42":-100},`+
		`"max_tokens":16,"messages":[{"role":"user","content":"Name a color."}]}`, stream)
	request, envelope, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIChatV1, []byte(body))
	require.NoError(t, err)
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, &request)
	ctx.ProtocolEnvelope = envelope
	ctx.ExpectStreamingResponse = stream
	ctx.StartTime, ctx.ProcessingStartTime = time.Now(), time.Now()
	decision := router.Config.GetDecisionByName(decisionName)
	require.NotNil(t, decision)
	ctx.VSRSelectedDecision = decision
	ctx.VSRSelectedDecisionName = decision.Name
	ctx.VSREligibleModelRefs = cloneModelRefs(decision.ModelRefs)
	return ctx
}

// sentSamplingFields reports which of min_p and logit_bias a provider call
// carries.
func sentSamplingFields(t *testing.T, body []byte) (minP, logitBias bool) {
	t.Helper()
	var sent map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(body, &sent), "%s", body)
	_, minP = sent["min_p"]
	_, logitBias = sent["logit_bias"]
	return minP, logitBias
}

func droppedSamplingWarning(field string) string {
	return `protocol;action=dropped;reason="the selected model does not support ` + field + `";field="` + field + `"`
}

func samplingResponseHeaders(contentType string) *ext_proc.ProcessingRequest_ResponseHeaders {
	return &ext_proc.ProcessingRequest_ResponseHeaders{ResponseHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":status", Value: "200"}, {Key: "content-type", Value: contentType},
		}},
	}}
}

// A request that carries min_p and logit_bias reaches the decision's model
// that runs speculative decoding without them, and the answer's
// x-vsr-warnings reports each, buffered or streamed. The decision's other
// model still receives both, and the client's request keeps them.
func TestSamplingControlsAreLeftOutOnlyForTheModelThatLacksThem(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, tc := range []struct {
			model string
			drops bool
		}{{model: "draft-model", drops: true}, {model: "plain-model"}} {
			t.Run(fmt.Sprintf("%s/stream=%t", tc.model, stream), func(t *testing.T) {
				router := samplingCapabilityRouter(t)
				ctx := samplingRequestContext(t, router, "chat", stream)
				_, err := router.prepareProviderDispatch(ctx.SemanticRequest, tc.model, "chat", false, ctx)
				require.NoError(t, err)
				body, err := router.encodeDispatchRequest(ctx)
				require.NoError(t, err)
				minP, logitBias := sentSamplingFields(t, body)
				require.Equal(t, !tc.drops, minP, "min_p in the call to %s: %s", tc.model, body)
				require.Equal(t, !tc.drops, logitBias, "logit_bias in the call to %s: %s", tc.model, body)
				client, err := protocolcodec.NewBuiltinEngine().EncodeRequest(llmprotocol.OpenAIChatV1, *ctx.SemanticRequest, llmprotocol.Envelope{})
				require.NoError(t, err)
				minP, logitBias = sentSamplingFields(t, client.Body)
				require.True(t, minP && logitBias, "the client's request lost min_p or logit_bias: %s", client.Body)

				contentType := "application/json"
				if stream {
					contentType = "text/event-stream"
				}
				response, err := router.handleResponseHeaders(samplingResponseHeaders(contentType), ctx)
				require.NoError(t, err)
				require.Equal(t, stream, ctx.IsStreamingResponse)
				warnings := protocolDiagnosticsHeader(response.GetResponseHeaders().GetResponse().GetHeaderMutation())
				for _, field := range []string{"min_p", "logit_bias"} {
					require.Equal(t, tc.drops, strings.Contains(warnings, droppedSamplingWarning(field)),
						"%s warning in x-vsr-warnings = %q", field, warnings)
				}
			})
		}
	}
}

// A decision that falls back between its two models sends each candidate the
// request it can serve, buffered or streamed. When plain-model fails,
// draft-model answers without min_p and logit_bias, which it would refuse,
// and the answer warns about both; when draft-model fails, plain-model still
// receives both.
func TestFallbackSendsEachCandidateTheSamplingControlsItSupports(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, tc := range []struct {
			primary, candidate string
			drops              bool
		}{
			{primary: "plain-model", candidate: "draft-model", drops: true},
			{primary: "draft-model", candidate: "plain-model"},
		} {
			t.Run(fmt.Sprintf("%s to %s/stream=%t", tc.primary, tc.candidate, stream), func(t *testing.T) {
				router := samplingCapabilityRouter(t)
				ctx := samplingRequestContext(t, router, "chat", stream)
				_, err := router.prepareProviderDispatch(ctx.SemanticRequest, tc.primary, "chat", false, ctx)
				require.NoError(t, err)
				_, err = router.encodeDispatchRequest(ctx)
				require.NoError(t, err)
				snapshot, err := cloneSemanticRequestForReplay(ctx.SemanticRequest)
				require.NoError(t, err)
				ctx.FallbackRequest = snapshot

				var sent []byte
				router.fallbackCaller = func(_ context.Context, model string, body []byte, _ map[string]string) ([]byte, int, error) {
					require.Equal(t, tc.candidate, model)
					sent = body
					if minP, logitBias := sentSamplingFields(t, body); model == "draft-model" && (minP || logitBias) {
						return []byte(speculativeDecodingRejection), http.StatusBadRequest, nil
					}
					return []byte(samplingCandidateAnswer), http.StatusOK, nil
				}
				response, err := router.handleResponseHeaders(status503(), ctx)
				require.NoError(t, err)
				require.NotNil(t, sent, "the candidate was never called")
				minP, logitBias := sentSamplingFields(t, sent)
				require.Equal(t, !tc.drops, minP, "min_p in the call to %s: %s", tc.candidate, sent)
				require.Equal(t, !tc.drops, logitBias, "logit_bias in the call to %s: %s", tc.candidate, sent)

				immediate := response.GetImmediateResponse()
				require.NotNil(t, immediate, "no fallback answer: %+v", response)
				require.Equal(t, typev3.StatusCode_OK, immediate.GetStatus().GetCode(), "%s", immediate.GetBody())
				require.NoError(t, protocolcodec.NewBuiltinEngine().ValidateEncodedResponse(immediate.GetBody(), stream))
				require.Contains(t, string(immediate.GetBody()), "Teal is a color.")
				require.Equal(t, tc.candidate, ctx.VSRSelectedModel)
				if tc.drops {
					warnings := immediateHeaderValue(response, headers.VSRWarnings)
					for _, field := range []string{"min_p", "logit_bias"} {
						require.Contains(t, warnings, droppedSamplingWarning(field))
					}
				}
			})
		}
	}
}

// A decision's request_params can block logit_bias for every model it routes
// to; the client's min_p still reaches a model that supports it.
func TestDecisionBlockedParamsRemoveLogitBias(t *testing.T) {
	router := samplingCapabilityRouter(t)
	ctx := samplingRequestContext(t, router, "blocked", false)
	_, err := router.prepareProviderDispatch(ctx.SemanticRequest, "plain-model", "blocked", false, ctx)
	require.NoError(t, err)
	body, err := router.encodeDispatchRequest(ctx)
	require.NoError(t, err)
	minP, logitBias := sentSamplingFields(t, body)
	require.True(t, minP, "min_p was removed: %s", body)
	require.False(t, logitBias, "logit_bias was not blocked: %s", body)
}
