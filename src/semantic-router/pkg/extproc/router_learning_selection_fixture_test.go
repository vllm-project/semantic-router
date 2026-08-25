/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func newRouterLearningEntrypointFixture(
	t *testing.T,
	cfg config.RouterConfig,
	assignments map[string][]string,
) (*OpenAIRouter, *config.RoutingRecipe) {
	t.Helper()
	if len(cfg.Recipes) != 1 {
		t.Fatalf("router learning fixture requires exactly one Recipe, got %d", len(cfg.Recipes))
	}
	recipe := &cfg.Recipes[0]
	recipe.ID = "recipe-router-learning"
	recipe.Revision = 1
	recipe.Name = "router-learning"

	actionAssignments := routerLearningFixtureAssignments(t, &cfg, assignments)
	cfg.Entrypoints = []config.EntrypointMapping{{
		ID:         "entrypoint-router-learning",
		Revision:   1,
		Name:       "router-learning",
		ModelNames: []string{"router/learning"},
		Rules: []config.EntrypointRule{{
			ID:   "rule-router-learning",
			Name: "router-learning",
			Action: config.EntrypointRuleAction{
				RecipeID:       recipe.ID,
				RecipeRevision: recipe.Revision,
				Recipe:         recipe.Name,
				Assignments:    actionAssignments,
			},
		}},
	}}
	if err := cfg.PrepareEntrypointRecipes(); err != nil {
		t.Fatalf("prepare router learning Entrypoint: %v", err)
	}
	compiled, ok := cfg.RecipeForRequestModel("router/learning")
	if !ok {
		t.Fatal("resolve router learning Entrypoint")
	}
	return &OpenAIRouter{Config: &cfg}, compiled
}

func routerLearningFixtureAssignments(
	t *testing.T,
	cfg *config.RouterConfig,
	assignments map[string][]string,
) map[string]config.RoutingAssignmentSet {
	t.Helper()
	result := make(map[string]config.RoutingAssignmentSet, len(assignments))
	for decisionID, modelNames := range assignments {
		models := make([]config.RoutingModelAssignment, 0, len(modelNames))
		for _, modelName := range modelNames {
			params, ok := cfg.ModelConfig[modelName]
			if !ok {
				t.Fatalf("router learning fixture references unknown model %q", modelName)
			}
			if params.ResourceID == "" {
				params.ResourceID = "model-" + modelName
				params.ResourceRevision = 1
				cfg.ModelConfig[modelName] = params
			}
			models = append(models, config.RoutingModelAssignment{
				ModelID: params.ResourceID, ModelRevision: params.ResourceRevision,
				ModelName: modelName, Weight: "1",
			})
		}
		result[decisionID] = config.RoutingAssignmentSet{Models: models}
	}
	return result
}
