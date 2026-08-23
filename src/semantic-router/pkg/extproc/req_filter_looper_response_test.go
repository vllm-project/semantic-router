package extproc

import (
	"context"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dispatchauthority"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func TestCreateLooperResponseIncludesTrackedHeaders(t *testing.T) {
	resp := &looper.Response{
		Semantic:      looperTestSemanticResponse(t),
		Model:         "model-b",
		ModelsUsed:    []string{"model-a", "model-b"},
		Iterations:    2,
		AlgorithmType: "elo",
	}
	// The looper trace, matched signals, category and session phase are demoted
	// to x-vsr-debug (#2205), so this tracked-headers test opts into debug.
	reqCtx := &RequestContext{
		Headers:                       map[string]string{headers.VSRDebug: "true"},
		VSRMatchedKeywords:            []string{"python"},
		VSRMatchedEmbeddings:          []string{"coding"},
		VSRMatchedContext:             []string{"memory"},
		VSRMatchedComplexity:          []string{"complexity:medium"},
		VSRMatchedModality:            []string{"AR"},
		VSRMatchedAuthz:               []string{"authz:team-a"},
		VSRMatchedJailbreak:           []string{"jailbreak:block"},
		VSRMatchedPII:                 []string{"pii:email"},
		VSRMatchedReask:               []string{"likely_dissatisfied"},
		VSRMatchedProjection:          []string{"balance_reasoning"},
		VSRContextTokenCount:          42,
		VSRSelectedModel:              "model-b",
		VSRSelectedDecisionName:       "coding",
		VSRSelectedDecisionConfidence: 0,
		VSRSelectedCategory:           "programming",
		RouterReplayID:                "replay-123",
		VSRLearningPolicies: testLearningPolicies(
			replayTestProtectionPolicyWithTrace(&selection.SessionPolicyTrace{
				Phase: "tool_loop",
			}),
		),
	}

	response := (&OpenAIRouter{}).createLooperResponse(resp, reqCtx)
	headerMap := headerValuesByName(response.GetImmediateResponse().Headers.SetHeaders)

	assert.Equal(t, "application/json", headerMap["content-type"])
	assert.Equal(t, "model-b", headerMap[headers.VSRLooperModel])
	assert.Equal(t, "model-a,model-b", headerMap[headers.VSRLooperModelsUsed])
	assert.Equal(t, "2", headerMap[headers.VSRLooperIterations])
	assert.Equal(t, "elo", headerMap[headers.VSRLooperAlgorithm])
	assert.Equal(t, "python", headerMap[headers.VSRMatchedKeywords])
	assert.Equal(t, "complexity:medium", headerMap[headers.VSRMatchedComplexity])
	assert.Equal(t, "AR", headerMap[headers.VSRMatchedModality])
	assert.Equal(t, "authz:team-a", headerMap[headers.VSRMatchedAuthz])
	assert.Equal(t, "jailbreak:block", headerMap[headers.VSRMatchedJailbreak])
	assert.Equal(t, "pii:email", headerMap[headers.VSRMatchedPII])
	assert.Equal(t, "likely_dissatisfied", headerMap[headers.VSRMatchedReask])
	assert.Equal(t, "balance_reasoning", headerMap[headers.VSRMatchedProjection])
	assert.Equal(t, "model-b", headerMap[headers.VSRSelectedModel])
	assert.Equal(t, "coding", headerMap[headers.VSRSelectedDecision])
	assert.Equal(t, "0.0000", headerMap[headers.VSRSelectedConfidence])
	assert.Equal(t, "programming", headerMap[headers.VSRSelectedCategory])
	assert.Equal(t, "tool_loop", headerMap[headers.VSRSessionPhase])
	assert.Equal(t, "replay-123", headerMap[headers.RouterReplayID])
	assert.Equal(t, "42", headerMap[headers.VSRContextTokenCount])
}

func TestCreateLooperResponseDefaultSurfaceIsLean(t *testing.T) {
	// Without x-vsr-debug the looper response carries only content-type, the
	// keystone headers and the final routing facts (#2205). The execution trace,
	// matched signals, category and session phase are demoted.
	resp := &looper.Response{
		Semantic:      looperTestSemanticResponse(t),
		Model:         "model-b",
		ModelsUsed:    []string{"model-a", "model-b"},
		Iterations:    2,
		AlgorithmType: "elo",
	}
	reqCtx := &RequestContext{
		VSRMatchedKeywords:            []string{"python"},
		VSRContextTokenCount:          42,
		VSRSelectedModel:              "model-b",
		VSRSelectedDecisionName:       "coding",
		VSRSelectedDecisionConfidence: 0,
		VSRSelectedCategory:           "programming",
		RouterReplayID:                "replay-123",
		VSRLearningPolicies: testLearningPolicies(
			replayTestProtectionPolicyWithTrace(&selection.SessionPolicyTrace{
				Phase: "tool_loop",
			}),
		),
	}

	response := (&OpenAIRouter{}).createLooperResponse(resp, reqCtx)
	headerMap := headerValuesByName(response.GetImmediateResponse().Headers.SetHeaders)

	// content-type, keystone and final routing facts ride on the default surface.
	assert.Equal(t, "application/json", headerMap["content-type"])
	assert.Equal(t, headers.SchemaVersionValue, headerMap[headers.VSRSchemaVersion])
	assert.Equal(t, headers.ResponsePathLooper, headerMap[headers.VSRResponsePath])
	assert.Equal(t, "model-b", headerMap[headers.VSRSelectedModel])
	assert.Equal(t, "coding", headerMap[headers.VSRSelectedDecision])
	assert.Equal(t, "0.0000", headerMap[headers.VSRSelectedConfidence])
	assert.Equal(t, "replay-123", headerMap[headers.RouterReplayID])

	// The demoted detail headers are absent.
	assert.NotContains(t, headerMap, headers.VSRLooperModel)
	assert.NotContains(t, headerMap, headers.VSRLooperIterations)
	assert.NotContains(t, headerMap, headers.VSRMatchedKeywords)
	assert.NotContains(t, headerMap, headers.VSRContextTokenCount)
	assert.NotContains(t, headerMap, headers.VSRSelectedCategory)
	assert.NotContains(t, headerMap, headers.VSRSessionPhase)
}

func looperTestChatResponseBody() []byte {
	return []byte(`{"id":"chatcmpl-looper","object":"chat.completion","created":1,"model":"model-b","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
}

func looperTestSemanticResponse(t testing.TB) *llmprotocol.Response {
	t.Helper()
	response, _, _, err := protocolcodec.NewBuiltinEngine().DecodeResponse(
		llmprotocol.OpenAIChatV1,
		looperTestChatResponseBody(),
	)
	require.NoError(t, err)
	return &response
}

func TestGetReasoningInfoFromDecision(t *testing.T) {
	useReasoning := true
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				ReasoningConfig: config.ReasoningConfig{DefaultReasoningEffort: "high"},
			},
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"model-a": {ReasoningFamily: "qwen3"},
					"model-b": {ReasoningFamily: "deepseek"},
				},
			},
		},
	}
	decision := &config.Decision{
		ModelRefs: []config.ModelRef{
			{
				Model: "model-a",
				ModelReasoningControl: config.ModelReasoningControl{
					UseReasoning:    &useReasoning,
					ReasoningEffort: "low",
				},
			},
		},
	}

	use, effort := router.getReasoningInfoFromDecision(decision, "model-a")
	assert.True(t, use)
	assert.Equal(t, "low", effort)

	use, effort = router.getReasoningInfoFromDecision(decision, "model-b")
	assert.True(t, use)
	assert.Equal(t, "high", effort)
}

func TestLooperDispatchIncludesPluginHeadersWithoutPhysicalRoutingMetadata(t *testing.T) {
	decision := &config.Decision{
		Name: "coding",
		Plugins: []config.DecisionPlugin{
			{
				Type: "header_mutation",
				Configuration: config.MustStructuredPayload(map[string]interface{}{
					"add": []map[string]interface{}{
						{"name": "x-extra", "value": "1"},
					},
					"delete": []string{"x-remove-me"},
				}),
			},
		},
	}
	router := newLooperDispatchTestRouter("model-a")
	ctx := looperDispatchTestContext()
	ctx.SourceFormat = llmprotocol.OpenAIChatV1
	ctx.SemanticRequest = testNeutralRequest("model-a", "hi")
	ctx.VSRSelectedDecision = decision
	response := router.buildLooperBackendDispatchResponse("model-a", []byte(`{"model":"model-a"}`), ctx)
	common := response.GetRequestBody().GetResponse()
	headerMap := headerValuesByName(common.GetHeaderMutation().GetSetHeaders())

	assert.Equal(t, "model-a", headerMap[headers.SelectedModel])
	assert.Equal(t, "model-a", headerMap[headers.VSRSelectedModel])
	assert.Equal(t, "capability", headerMap[backendinvoker.DispatchCapabilityHeader])
	assert.Empty(t, headerMap["authorization"])
	assert.Empty(t, headerMap["x-vsr-destination-endpoint"])
	assert.Equal(t, "1", headerMap["x-extra"])
	assert.Contains(t, common.GetHeaderMutation().GetRemoveHeaders(), "content-length")
	assert.Contains(t, common.GetHeaderMutation().GetRemoveHeaders(), headers.VSRSelectedRecipe)
	assert.Contains(t, common.GetHeaderMutation().GetRemoveHeaders(), "x-remove-me")
}

func TestHandleLooperInternalRequestRewritesModel(t *testing.T) {
	router := newLooperDispatchTestRouter("model-b")
	ctx := looperDispatchTestContext()
	ctx.SourceFormat = llmprotocol.OpenAIChatV1
	ctx.SemanticRequest = testNeutralRequest("auto", "hi")
	ctx.LooperRequest = true

	response, err := router.handleLooperInternalRequest("model-b", ctx)
	require.NoError(t, err)
	require.NotNil(t, response.GetRequestBody())

	body := response.GetRequestBody().Response.GetBodyMutation().GetBody()
	assert.JSONEq(
		t,
		`{"model":"model-b","messages":[{"role":"user","content":"hi"}]}`,
		string(body),
	)

	headerMap := headerValuesByName(response.GetRequestBody().Response.HeaderMutation.SetHeaders)
	assert.Equal(t, "model-b", headerMap[headers.SelectedModel])
	assert.Equal(t, "model-b", headerMap[headers.VSRSelectedModel])
	assert.Equal(t, "capability", headerMap[backendinvoker.DispatchCapabilityHeader])
	assert.Contains(t, response.GetRequestBody().Response.HeaderMutation.RemoveHeaders, "content-length")
}

func TestHandleLooperInternalRequestWithPluginsKeepsLogicalModel(t *testing.T) {
	router := newLooperDispatchTestRouter("panel-a")
	router.Cache = &spyCache{}
	router.Config.Recipes = []config.RoutingRecipe{{
		Name: "fusion-recipe",
		Profile: config.RoutingProfile{
			Decisions: []config.Decision{{
				Name: "fusion_alias", ModelRefs: []config.ModelRef{{Model: "panel-a"}},
			}},
		},
	}}
	ctx := looperDispatchTestContext()
	ctx.LooperRequest = true
	ctx.Headers = map[string]string{
		headers.VSRLooperDecision: "fusion_alias",
		headers.VSRSelectedRecipe: "fusion-recipe",
	}
	ctx.SourceFormat = llmprotocol.OpenAIChatV1
	ctx.SemanticRequest = testNeutralRequest("panel-a", "hi")

	response, err := router.handleLooperInternalRequestWithPlugins("panel-a", ctx)
	require.NoError(t, err)
	require.NotNil(t, response.GetRequestBody())

	body := response.GetRequestBody().Response.GetBodyMutation().GetBody()
	assert.Contains(t, string(body), `"model":"panel-a"`)

	headerMap := headerValuesByName(response.GetRequestBody().Response.HeaderMutation.SetHeaders)
	assert.Equal(t, "panel-a", headerMap[headers.SelectedModel])
	assert.Equal(t, "panel-a", headerMap[headers.VSRSelectedModel])
	assert.Equal(t, "capability", headerMap[backendinvoker.DispatchCapabilityHeader])
}

func newLooperDispatchTestRouter(model string) *OpenAIRouter {
	return &OpenAIRouter{
		Config: &config.RouterConfig{BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				model: {ResourceID: "model-id", ResourceRevision: 1},
			},
		}},
		DispatchCapabilities: dispatchCapabilityRuntimeStub{},
	}
}

func looperDispatchTestContext() *RequestContext {
	return &RequestContext{
		Headers: map[string]string{},
		TraceContext: withVerifiedDispatchGrant(
			context.Background(), dispatchauthority.VerifiedGrant{},
		),
	}
}

func headerValuesByName(headers []*core.HeaderValueOption) map[string]string {
	result := make(map[string]string, len(headers))
	for _, header := range headers {
		result[header.Header.Key] = string(header.Header.RawValue)
	}
	return result
}
