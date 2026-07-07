//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestOpenAIModelsEndpoint(t *testing.T) {
	tests := []struct {
		name                      string
		includeConfiguredModels   bool
		config                    *config.RouterConfig
		expectedModels            []string
		expectedModelResultLength int
	}{
		{
			name:                      "default excludes config models",
			includeConfiguredModels:   false,
			expectedModels:            []string{"vllm-sr/auto", "auto", "MoM"},
			expectedModelResultLength: 3,
		},
		{
			name:                      "router option includes config models",
			includeConfiguredModels:   true,
			expectedModels:            []string{"vllm-sr/auto", "auto", "MoM", "gpt-4o-mini", "llama-3.1-8b-instruct"},
			expectedModelResultLength: 5,
		},
		{
			name:   "direct looper models are exposed when decisions are configured",
			config: openAIModelsLooperTestConfig(),
			expectedModels: []string{
				"vllm-sr/auto",
				"auto",
				"MoM",
				"vllm-sr/remom",
				"vllm-sr/fusion",
				"vllm-sr/flow",
			},
			expectedModelResultLength: 6,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			apiServer := &ClassificationAPIServer{
				classificationSvc: services.NewPlaceholderClassificationService(),
				config:            openAIModelsTestConfigForCase(tt.config, tt.includeConfiguredModels),
			}

			req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
			rr := httptest.NewRecorder()
			apiServer.handleOpenAIModels(rr, req)

			if rr.Code != http.StatusOK {
				t.Fatalf("expected 200 OK, got %d", rr.Code)
			}

			var resp OpenAIModelList
			if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
				t.Fatalf("failed to parse response: %v", err)
			}

			assertOpenAIModelList(t, resp, tt.expectedModels, tt.expectedModelResultLength)
		})
	}
}

func openAIModelsTestConfigForCase(cfg *config.RouterConfig, includeConfiguredModels bool) *config.RouterConfig {
	if cfg != nil {
		return cfg
	}
	return openAIModelsTestConfig(includeConfiguredModels)
}

func openAIModelsTestConfig(includeConfiguredModels bool) *config.RouterConfig {
	return &config.RouterConfig{
		BackendModels: config.BackendModels{
			VLLMEndpoints: []config.VLLMEndpoint{
				{
					Name:    "primary",
					Address: "127.0.0.1",
					Port:    8000,
					Weight:  1,
				},
			},
			ModelConfig: map[string]config.ModelParams{
				"gpt-4o-mini": {
					PreferredEndpoints: []string{"primary"},
				},
				"llama-3.1-8b-instruct": {
					PreferredEndpoints: []string{"primary"},
				},
			},
		},
		RouterOptions: config.RouterOptions{
			IncludeConfigModelsInList: includeConfiguredModels,
		},
	}
}

func openAIModelsLooperTestConfig() *config.RouterConfig {
	return &config.RouterConfig{
		Looper: config.LooperConfig{Endpoint: "http://looper"},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name:      "remom-route",
					ModelRefs: []config.ModelRef{{Model: "worker-a"}},
					Algorithm: &config.AlgorithmConfig{
						Type:  "remom",
						ReMoM: &config.ReMoMAlgorithmConfig{BreadthSchedule: []int{1}},
					},
				},
				{
					Name:      "fusion-route",
					ModelRefs: []config.ModelRef{{Model: "worker-a"}},
					Algorithm: &config.AlgorithmConfig{
						Type: "fusion",
					},
				},
				{
					Name:      "flow-route",
					ModelRefs: []config.ModelRef{{Model: "worker-a"}},
					Algorithm: &config.AlgorithmConfig{
						Type: "workflows",
					},
				},
			},
		},
	}
}

func assertOpenAIModelList(t *testing.T, resp OpenAIModelList, expectedModels []string, expectedLength int) {
	t.Helper()

	if resp.Object != "list" {
		t.Fatalf("expected object 'list', got %s", resp.Object)
	}

	got := map[string]bool{}
	for _, model := range resp.Data {
		got[model.ID] = true
		if model.Object != "model" {
			t.Fatalf("expected each item.object to be 'model', got %s", model.Object)
		}
		if model.Created == 0 {
			t.Fatalf("expected created timestamp to be non-zero")
		}
	}

	for _, model := range expectedModels {
		if !got[model] {
			t.Fatalf("expected list to contain %q, got: %v", model, got)
		}
	}
	if len(resp.Data) != expectedLength {
		t.Fatalf("expected %d models, got %d: %v", expectedLength, len(resp.Data), got)
	}
}
