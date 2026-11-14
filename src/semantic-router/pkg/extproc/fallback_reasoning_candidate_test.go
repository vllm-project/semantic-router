package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// thinkingFamily switches thinking on and off and defaults to on, like the
// built-in catalog's Qwen3.8 and GLM families.
var thinkingFamily = config.ReasoningFamilyConfig{
	Type:                config.ReasoningFamilyTypeReasoningEffort,
	Parameter:           "reasoning_effort",
	ActivationParameter: "enable_thinking",
	Levels:              []string{"low", "medium", "high"},
	Default:             "high",
	Modes:               []string{"enabled", "disabled"},
	DefaultMode:         "enabled",
}

// withThinkingCandidate makes candidate the decision's second model with
// reasoning on at effort, and takes the primary's fallback snapshot the way
// the body phase takes it before the first call.
func withThinkingCandidate(t *testing.T, router *OpenAIRouter, ctx *RequestContext, candidate, effort string) {
	t.Helper()
	cfg := router.Config
	cfg.ReasoningFamilies = map[string]config.ReasoningFamilyConfig{"thinking": thinkingFamily}
	params := cfg.ModelConfig[candidate]
	params.ReasoningFamily = "thinking"
	cfg.ModelConfig[candidate] = params
	if candidate == "model-fallback-1" {
		cfg.ProviderProfiles["prof-vllm"] = config.ProviderProfile{Type: "vllm", BaseURL: "http://127.0.0.1:8002"}
		cfg.VLLMEndpoints[1].ProviderProfileName = "prof-vllm"
	}
	on := true
	ctx.VSREligibleModelRefs = []config.ModelRef{
		{Model: "model-primary"},
		{Model: candidate, ModelReasoningControl: config.ModelReasoningControl{UseReasoning: &on, ReasoningEffort: effort}},
	}
	ctx.VSRSelectedDecision = &config.Decision{Name: ctx.VSRSelectedDecisionName, ModelRefs: ctx.VSREligibleModelRefs}
	snapshot, err := cloneSemanticRequestForReplay(ctx.SemanticRequest)
	require.NoError(t, err)
	ctx.FallbackRequest = snapshot
}

func streamRequested(ctx *RequestContext) {
	ctx.ExpectStreamingResponse = true
	ctx.SemanticRequest.Stream = true
}

func messagesClient(ctx *RequestContext) {
	ctx.SourceFormat = llmprotocol.AnthropicMessagesV1
	ctx.SemanticRequest.Trusted.SourceFormat = llmprotocol.AnthropicMessagesV1
}

func thinkingKwargs(t *testing.T, body []byte) map[string]interface{} {
	t.Helper()
	var request map[string]interface{}
	require.NoError(t, json.Unmarshal(body, &request))
	kwargs, ok := request["chat_template_kwargs"].(map[string]interface{})
	require.True(t, ok, "no chat_template_kwargs in %s", body)
	return kwargs
}

// A second model with reasoning on answers after the primary's 503 in the
// native gateway, whether the client streams or not and whichever protocol it
// speaks: the candidate's call carries its backend's own reasoning control.
func TestNativeFallbackSendsACandidateWithReasoningOn(t *testing.T) {
	for _, client := range []string{"chat", "messages"} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%v", client, stream), func(t *testing.T) {
				session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), "model-primary", "model-fallback-1")
				ctx := session.ctx
				if client == "messages" {
					messagesClient(ctx)
				}
				if stream {
					streamRequested(ctx)
				}
				withThinkingCandidate(t, session.router, ctx, "model-fallback-1", "medium")

				step, err := session.Fallback().Next(context.Background(), failedWith(http.StatusServiceUnavailable, "unavailable"))
				require.NoError(t, err)
				require.NotNil(t, step.Call, "the candidate with reasoning on was skipped")
				require.Equal(t, "model-fallback-1", step.Call.Route)
				kwargs := thinkingKwargs(t, step.Call.Request.Body)
				require.Equal(t, true, kwargs["enable_thinking"])
				require.Equal(t, "medium", kwargs["reasoning_effort"])
				require.Equal(t, stream, strings.Contains(string(step.Call.Request.Body), `"stream":true`))

				session.fallback.succeeded(http.StatusOK)
				require.Equal(t, "model-fallback-1", ctx.VSRSelectedModel)
				require.Equal(t, "succeeded", ctx.FallbackRecord.FinalStatus)
				require.Len(t, ctx.FallbackRecord.Attempts, 2)
			})
		}
	}
}

// The response-phase fallback behind Envoy answers with the same candidate,
// buffered or re-encoded as the client's stream.
func TestResponsePhaseFallbackSendsACandidateWithReasoningOn(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
			router, _ := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
			ctx := testFallbackRequestContext("model-primary", []string{"model-primary", "model-fallback-1"})
			ctx.UpstreamStatusCode = http.StatusServiceUnavailable
			if stream {
				streamRequested(ctx)
			}
			withThinkingCandidate(t, router, ctx, "model-fallback-1", "medium")

			var sent []byte
			router.fallbackCaller = func(_ context.Context, model string, body []byte, _ map[string]string) ([]byte, int, error) {
				require.Equal(t, "model-fallback-1", model)
				sent = append([]byte(nil), body...)
				return []byte(`{"id":"chatcmpl-thinking","object":"chat.completion","created":1700000000,` +
					`"model":"model-fallback-1","choices":[{"index":0,"message":{"role":"assistant",` +
					`"content":"from the second model"},"finish_reason":"stop"}]}`), http.StatusOK, nil
			}

			resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
			require.NotNil(t, sent, "the candidate with reasoning on was never called")
			kwargs := thinkingKwargs(t, sent)
			require.Equal(t, true, kwargs["enable_thinking"])
			require.Equal(t, "medium", kwargs["reasoning_effort"])
			immediate := resp.GetImmediateResponse()
			require.NotNil(t, immediate)
			require.Equal(t, typev3.StatusCode_OK, immediate.GetStatus().GetCode())
			require.Contains(t, string(immediate.GetBody()), "from the second model")
			require.NoError(t, protocolcodec.NewBuiltinEngine().ValidateEncodedResponse(immediate.GetBody(), stream))
			require.Equal(t, "succeeded", ctx.FallbackRecord.FinalStatus)
		})
	}
}

// A Messages backend is the one wire with a reasoning mode: the candidate
// still gets its thinking, adaptive where it has no budget, as the primary.
func TestFallbackCandidateOnMessagesKeepsItsThinking(t *testing.T) {
	session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), "model-primary", "model-fallback-2")
	withThinkingCandidate(t, session.router, session.ctx, "model-fallback-2", "medium")

	step, err := session.Fallback().Next(context.Background(), failedWith(http.StatusServiceUnavailable, "unavailable"))
	require.NoError(t, err)
	require.NotNil(t, step.Call)
	require.Equal(t, "model-fallback-2", step.Call.Route)
	var request map[string]interface{}
	require.NoError(t, json.Unmarshal(step.Call.Request.Body, &request))
	thinking, ok := request["thinking"].(map[string]interface{})
	require.True(t, ok, "no thinking in %s", step.Call.Request.Body)
	require.Equal(t, "adaptive", thinking["type"])
}
