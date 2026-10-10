package extproc

import (
	"strings"
	"testing"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
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

	response, err := router.buildLooperBackendDispatchResponse(model, "", false, ctx)
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

// A credential failure on a Looper hop must reach the client with the status and
// message the dispatch builder produced, not a generic 500 (issue #4581).
func TestLooperProviderDispatchKeepsCredentialFailure(t *testing.T) {
	tests := []struct {
		name    string
		prepare func(router *OpenAIRouter)
		status  typev3.StatusCode
		message string
	}{
		{
			name:    "credential resolver unavailable",
			prepare: func(router *OpenAIRouter) { router.CredentialResolver = nil },
			status:  typev3.StatusCode_InternalServerError,
			message: "Provider credentials are unavailable.",
		},
		{
			name: "credential resolution fails",
			prepare: func(router *OpenAIRouter) {
				router.CredentialResolver = authz.NewCredentialResolver(authz.NewStaticConfigProvider(router.Config))
			},
			status:  typev3.StatusCode_Unauthorized,
			message: "Authentication failed. Check your API key configuration.",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
			tt.prepare(router)
			ctx := routingTestContext(llmprotocol.OpenAIChatV1, testNeutralRequest(model, "compare these answers"))
			ctx.LooperRequest = true

			looper, err := router.buildLooperBackendDispatchResponse(model, "", false, ctx)
			if err != nil {
				t.Fatalf("buildLooperBackendDispatchResponse: %v", err)
			}
			immediate := looper.GetImmediateResponse()
			if immediate == nil {
				t.Fatalf("Looper hop did not return the credential failure: %+v", looper)
			}
			if immediate.GetStatus().GetCode() != tt.status {
				t.Fatalf("status = %v, want %v", immediate.GetStatus().GetCode(), tt.status)
			}
			if !strings.Contains(string(immediate.GetBody()), tt.message) {
				t.Fatalf("body = %s, want message %q", immediate.GetBody(), tt.message)
			}

			// The ordinary dispatch path reports the same failure.
			dispatch, err := router.prepareProviderDispatch(ctx.SemanticRequest, model, "", false, ctx)
			if err != nil {
				t.Fatalf("prepareProviderDispatch: %v", err)
			}
			ordinary := router.buildProviderDispatchResponse(dispatch, ctx).GetImmediateResponse()
			if ordinary.GetStatus().GetCode() != immediate.GetStatus().GetCode() ||
				string(ordinary.GetBody()) != string(immediate.GetBody()) {
				t.Fatalf("Looper failure %v %s differs from ordinary %v %s",
					immediate.GetStatus().GetCode(), immediate.GetBody(), ordinary.GetStatus().GetCode(), ordinary.GetBody())
			}
		})
	}
}
