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
			name:                      "empty routing config exposes no implicit models",
			includeConfiguredModels:   false,
			expectedModels:            nil,
			expectedModelResultLength: 0,
		},
		{
			name:                      "router option includes config models",
			includeConfiguredModels:   true,
			expectedModels:            []string{"gpt-4o-mini", "llama-3.1-8b-instruct"},
			expectedModelResultLength: 2,
		},
		{
			name:   "entrypoint model names are exposed",
			config: openAIModelsEntrypointTestConfig(t),
			expectedModels: []string{
				"vllm-sr/privacy",
				"vllm-sr/default-alias",
			},
			expectedModelResultLength: 2,
		},
		{
			name:                      "unpublished orchestration recipes do not create model aliases",
			config:                    openAIModelsLooperTestConfig(),
			expectedModels:            nil,
			expectedModelResultLength: 0,
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

func openAIModelsEntrypointTestConfig(t *testing.T) *config.RouterConfig {
	t.Helper()
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"backend": {ResourceID: "model-backend", ResourceRevision: 1},
		}},
		Recipes: []config.RoutingRecipe{
			{
				ID: "recipe-default", Revision: 1, Name: config.DefaultRecipeName,
				Profile: config.RoutingProfile{Decisions: []config.Decision{{ID: "decision-default", Name: "default"}}},
			},
			{
				ID: "recipe-privacy", Revision: 1, Name: "privacy", Description: "privacy profile",
				Profile: config.RoutingProfile{Decisions: []config.Decision{{ID: "decision-privacy", Name: "privacy"}}},
			},
		},
		Entrypoints: []config.EntrypointMapping{
			{
				ID: "entrypoint-privacy", Revision: 1, Name: "privacy", ModelNames: []string{"vllm-sr/privacy"},
				Rules: []config.EntrypointRule{{
					ID: "rule-privacy", Name: "default",
					Action: config.EntrypointRuleAction{
						RecipeID: "recipe-privacy", RecipeRevision: 1, Recipe: "privacy",
						Assignments: map[string]config.RoutingAssignmentSet{
							"decision-privacy": {Models: []config.RoutingModelAssignment{{ModelID: "model-backend", ModelRevision: 1, ModelName: "backend", Weight: "1"}}},
						},
					},
				}},
			},
			{
				ID: "entrypoint-default", Revision: 1, Name: "default", ModelNames: []string{"vllm-sr/default-alias"},
				Rules: []config.EntrypointRule{{
					ID: "rule-default", Name: "default",
					Action: config.EntrypointRuleAction{
						RecipeID: "recipe-default", RecipeRevision: 1, Recipe: config.DefaultRecipeName,
						Assignments: map[string]config.RoutingAssignmentSet{
							"decision-default": {Models: []config.RoutingModelAssignment{{ModelID: "model-backend", ModelRevision: 1, ModelName: "backend", Weight: "1"}}},
						},
					},
				}},
			},
		},
	}
	if err := cfg.PrepareEntrypointRecipes(); err != nil {
		t.Fatalf("prepare model-list Entrypoints: %v", err)
	}
	return cfg
}

func openAIModelsLooperTestConfig() *config.RouterConfig {
	return &config.RouterConfig{
		Looper: config.LooperConfig{Endpoint: "http://looper"},
		Recipes: []config.RoutingRecipe{{
			ID: "recipe-unpublished", Revision: 1, Name: "unpublished",
			Profile: config.RoutingProfile{Decisions: []config.Decision{
				{
					ID: "decision-remom", Name: "remom-route",
					Algorithm: &config.AlgorithmConfig{
						Type:  "remom",
						ReMoM: &config.ReMoMAlgorithmConfig{BreadthSchedule: []int{1}},
					},
				},
				{
					ID: "decision-fusion", Name: "fusion-route",
					Algorithm: &config.AlgorithmConfig{
						Type: "fusion",
					},
				},
				{
					ID: "decision-flow", Name: "flow-route",
					Algorithm: &config.AlgorithmConfig{
						Type: "workflows",
					},
				},
			}},
		}},
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
		if model.Routing.Resolution == "" {
			t.Fatalf("expected %s to declare routing metadata", model.ID)
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
