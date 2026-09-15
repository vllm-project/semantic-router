package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
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
			Algorithm: &config.AlgorithmConfig{Type: "router_dc"},
		}

		assert.False(t, router.shouldUseLooper(decision))
	})

	t.Run("ignores selection algorithms", func(t *testing.T) {
		router := &OpenAIRouter{
			Config: &config.RouterConfig{Looper: config.LooperConfig{Endpoint: "http://looper"}},
		}
		selectionAlgorithms := []string{"static", "router_dc", "automix", "hybrid", "knn", "kmeans", "svm", "mlp", "multi_factor", "latency_aware"}

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

	t.Run("allows fusion with algorithm analysis models", func(t *testing.T) {
		router := &OpenAIRouter{
			Config: &config.RouterConfig{Looper: config.LooperConfig{Endpoint: "http://looper"}},
		}
		decision := &config.Decision{
			Name: "fusion",
			Algorithm: &config.AlgorithmConfig{
				Type: "fusion",
				Fusion: &config.FusionAlgorithmConfig{
					Model:          "judge",
					AnalysisModels: []string{"model-a"},
				},
			},
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
			Algorithm: &config.AlgorithmConfig{Type: "router_dc"},
		}

		assert.False(t, router.shouldUseLooper(decision))
	})

	t.Run("allows looper-only algorithms with multiple models", func(t *testing.T) {
		router := &OpenAIRouter{
			Config: &config.RouterConfig{Looper: config.LooperConfig{Endpoint: "http://looper"}},
		}
		for _, algorithmType := range config.SupportedLooperAlgorithmTypes() {
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

func TestLooperProviderDispatchReevaluatesOntoSelectedRoute(t *testing.T) {
	router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
	router.Config.ClearRouteCache = true
	request := testNeutralRequest(model, "compare these answers")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.LooperRequest = true

	response, err := router.buildLooperBackendDispatchResponse(model, ctx)
	if err != nil {
		t.Fatalf("buildLooperBackendDispatchResponse: %v", err)
	}
	common := response.GetRequestBody().GetResponse()
	if !common.GetClearRouteCache() {
		t.Fatal("Looper provider dispatch did not clear the fallback route cache")
	}
	setHeaders := headerValuesByName(common.GetHeaderMutation().GetSetHeaders())
	if got := setHeaders[headers.SelectedModel]; got != model {
		t.Fatalf("selected model header = %q, want %q", got, model)
	}
	if got := setHeaders[":path"]; got != "/v1/chat/completions" {
		t.Fatalf("provider path = %q, want /v1/chat/completions", got)
	}
}
