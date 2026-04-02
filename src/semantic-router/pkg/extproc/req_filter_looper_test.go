package extproc

import (
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
)

func TestShouldUseLooper(t *testing.T) {
	t.Run("requires configured looper endpoint", func(t *testing.T) {
		router := &OpenAIRouter{Config: &config.RouterConfig{}}
		decision := &config.Decision{
			Name: "coding",
			ModelRefs: []config.ModelRef{
				{Model: "model-a"},
				{Model: "model-b"},
			},
			Algorithm: &config.AlgorithmConfig{Type: "elo"},
		}

		assert.False(t, router.shouldUseLooper(decision))
	})

	t.Run("ignores selection algorithms", func(t *testing.T) {
		router := &OpenAIRouter{
			Config: &config.RouterConfig{Looper: config.LooperConfig{Endpoint: "http://looper"}},
		}
		selectionAlgorithms := []string{"static", "elo", "router_dc", "automix", "hybrid", "knn", "kmeans", "svm", "mlp", "gmtrouter", "latency_aware"}

		for _, algorithmType := range selectionAlgorithms {
			decision := &config.Decision{
				Name: "routing",
				ModelRefs: []config.ModelRef{
					{Model: "model-a"},
					{Model: "model-b"},
				},
				Algorithm: &config.AlgorithmConfig{Type: algorithmType},
			}

			assert.False(t, router.shouldUseLooper(decision), "algorithm %s should use selector routing, not looper", algorithmType)
		}
	})

	t.Run("allows remom with single model", func(t *testing.T) {
		router := &OpenAIRouter{
			Config: &config.RouterConfig{Looper: config.LooperConfig{Endpoint: "http://looper"}},
		}
		decision := &config.Decision{
			Name:      "reasoning",
			ModelRefs: []config.ModelRef{{Model: "model-a"}},
			Algorithm: &config.AlgorithmConfig{Type: "remom"},
		}

		assert.True(t, router.shouldUseLooper(decision))
	})

	t.Run("requires multiple models for non-remom algorithms", func(t *testing.T) {
		router := &OpenAIRouter{
			Config: &config.RouterConfig{Looper: config.LooperConfig{Endpoint: "http://looper"}},
		}
		decision := &config.Decision{
			Name:      "routing",
			ModelRefs: []config.ModelRef{{Model: "model-a"}},
			Algorithm: &config.AlgorithmConfig{Type: "elo"},
		}

		assert.False(t, router.shouldUseLooper(decision))
	})

	t.Run("allows looper-only algorithms with multiple models", func(t *testing.T) {
		router := &OpenAIRouter{
			Config: &config.RouterConfig{Looper: config.LooperConfig{Endpoint: "http://looper"}},
		}
		looperAlgorithms := []string{"confidence", "ratings"}

		for _, algorithmType := range looperAlgorithms {
			decision := &config.Decision{
				Name: "routing",
				ModelRefs: []config.ModelRef{
					{Model: "model-a"},
					{Model: "model-b"},
				},
				Algorithm: &config.AlgorithmConfig{Type: algorithmType},
			}

			assert.True(t, router.shouldUseLooper(decision), "algorithm %s should use looper routing", algorithmType)
		}
	})
}

func TestCreateLooperResponseIncludesTrackedHeaders(t *testing.T) {
	resp := &looper.Response{
		Body:          []byte(`{"ok":true}`),
		ContentType:   "application/json",
		Model:         "model-b",
		ModelsUsed:    []string{"model-a", "model-b"},
		Iterations:    2,
		AlgorithmType: "elo",
	}
	reqCtx := &RequestContext{
		VSRMatchedKeywords:      []string{"python"},
		VSRMatchedEmbeddings:    []string{"coding"},
		VSRMatchedContext:       []string{"memory"},
		VSRMatchedComplexity:    []string{"complexity:medium"},
		VSRMatchedModality:      []string{"AR"},
		VSRMatchedAuthz:         []string{"authz:team-a"},
		VSRMatchedJailbreak:     []string{"jailbreak:block"},
		VSRMatchedPII:           []string{"pii:email"},
		VSRMatchedReask:         []string{"likely_dissatisfied"},
		VSRMatchedProjection:    []string{"balance_reasoning"},
		VSRContextTokenCount:    42,
		VSRSelectedDecisionName: "coding",
		VSRSelectedCategory:     "programming",
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
	assert.Equal(t, "coding", headerMap[headers.VSRSelectedDecision])
	assert.Equal(t, "programming", headerMap[headers.VSRSelectedCategory])
	assert.Equal(t, "42", headerMap[headers.VSRContextTokenCount])
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

func TestBuildHeaderMutationsForLooperIncludesAuthorizationAndPluginHeaders(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"model-a": {AccessKey: "secret"},
				},
			},
		},
	}
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

	setHeaders, removeHeaders := router.buildHeaderMutationsForLooper(decision, "model-a")
	headerMap := headerValuesByName(setHeaders)

	assert.Equal(t, "model-a", headerMap[headers.VSRSelectedModel])
	assert.Equal(t, "Bearer secret", headerMap["Authorization"])
	assert.Equal(t, "1", headerMap["x-extra"])
	assert.Contains(t, removeHeaders, "content-length")
	assert.Contains(t, removeHeaders, "x-remove-me")
}

func TestHandleLooperInternalRequestRewritesModel(t *testing.T) {
	ctx := &RequestContext{
		OriginalRequestBody: []byte(`{"model":"auto","messages":[{"role":"user","content":"hi"}]}`),
	}

	response, err := (&OpenAIRouter{}).handleLooperInternalRequest("model-b", ctx)
	require.NoError(t, err)
	require.NotNil(t, response.GetRequestBody())

	body := response.GetRequestBody().Response.GetBodyMutation().GetBody()
	assert.JSONEq(
		t,
		`{"model":"model-b","messages":[{"role":"user","content":"hi"}]}`,
		string(body),
	)
}

func headerValuesByName(headers []*core.HeaderValueOption) map[string]string {
	result := make(map[string]string, len(headers))
	for _, header := range headers {
		result[header.Header.Key] = string(header.Header.RawValue)
	}
	return result
}
