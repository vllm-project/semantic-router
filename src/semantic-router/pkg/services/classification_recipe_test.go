package services

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestEvalDecisionCandidatesSelectsEntrypointRecipe(t *testing.T) {
	const speedRecipe config.RecipeName = "speed-first"
	const (
		recipeID   = "rcp-speed"
		decisionID = "dec-flash"
		modelID    = "mdl-flash"
		modelName  = "backend/flash"
	)

	routerConfig := &config.RouterConfig{
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			modelName: {ResourceID: modelID, ResourceRevision: 1},
		}},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{Name: "balanced_route"}},
		},
		Entrypoints: []config.EntrypointMapping{
			{
				ID: "ep-speed", Revision: 1, Name: "Speed", ModelNames: []string{"router/speed-flash"},
				Rules: []config.EntrypointRule{{
					ID: "rule-default", Name: "Default",
					Action: config.EntrypointRuleAction{
						RecipeID: recipeID, RecipeRevision: 1, Recipe: speedRecipe,
						Assignments: map[string]config.RoutingAssignmentSet{
							decisionID: {Models: []config.RoutingModelAssignment{{
								ModelID: modelID, ModelRevision: 1, ModelName: modelName, Weight: "1",
							}}},
						},
					},
				}},
			},
		},
		Recipes: []config.RoutingRecipe{
			{Name: config.DefaultRecipeName, Profile: config.RoutingProfile{Decisions: []config.Decision{{Name: "balanced_route"}}}},
			{ID: recipeID, Revision: 1, Name: speedRecipe, Profile: config.RoutingProfile{Decisions: []config.Decision{{ID: decisionID, Name: "flash_route"}}}},
		},
	}
	require.NoError(t, routerConfig.PrepareEntrypointRecipes())
	service := &ClassificationService{config: routerConfig}

	_, candidates, recipe, runtimeScope, err := service.evalRoutingScope("router/speed-flash")
	require.NoError(t, err)
	require.Len(t, candidates, 1)
	assert.Equal(t, "flash_route", candidates[0].Name)
	assert.Equal(t, speedRecipe, recipe)
	assert.NotEmpty(t, runtimeScope)
	assert.NotEqual(t, recipe, runtimeScope)

	_, _, _, _, err = service.evalRoutingScope("router/missing")
	require.ErrorIs(t, err, ErrUnknownRoutingModel)

	response, err := service.ClassifyIntentForEval(IntentRequest{
		Text:  "hello",
		Model: "router/speed-flash",
	})
	require.NoError(t, err)
	assert.Equal(t, speedRecipe, response.Recipe)

	_, err = service.ClassifyIntentForEval(IntentRequest{
		Text:  "hello",
		Model: "router/missing",
	})
	require.ErrorIs(t, err, ErrUnknownRoutingModel)
}

func TestRecipeClassificationServiceRejectsConcreteBackendModel(t *testing.T) {
	routerConfig := &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{"backend-model": {}},
		},
		Recipes: []config.RoutingRecipe{{
			Name: config.DefaultRecipeName,
		}},
	}
	classifiers, err := classification.BuildRecipeClassifiers(routerConfig, nil, nil, nil)
	require.NoError(t, err)
	service := NewRecipeClassificationService(classifiers, routerConfig)

	_, err = service.ClassifyIntent(IntentRequest{Text: "hello", Model: "backend-model"})
	require.ErrorIs(t, err, ErrUnknownRoutingModel)
	_, err = service.ClassifyIntent(IntentRequest{Text: "hello", Model: "auto"})
	require.ErrorIs(t, err, ErrUnknownRoutingModel)

	classifier, err := service.classifierForRequestModel("")
	require.NoError(t, err)
	require.NotNil(t, classifier)
}

func TestRecipeClassificationServiceRefreshesNamedRecipePolicy(t *testing.T) {
	recipeConfig := func(expected string) *config.RouterConfig {
		const (
			recipeID   = "rcp-private"
			decisionID = "dec-private"
			modelID    = "mdl-private"
			modelName  = "backend/private"
		)
		result := &config.RouterConfig{
			BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
				modelName: {ResourceID: modelID, ResourceRevision: 1},
			}},
			Entrypoints: []config.EntrypointMapping{{
				ID: "ep-private", Revision: 1, Name: "Private", ModelNames: []string{"router/private"},
				Rules: []config.EntrypointRule{{
					ID: "rule-default", Name: "Default",
					Action: config.EntrypointRuleAction{
						RecipeID: recipeID, RecipeRevision: 1, Recipe: "private",
						Assignments: map[string]config.RoutingAssignmentSet{
							decisionID: {Models: []config.RoutingModelAssignment{{
								ModelID: modelID, ModelRevision: 1, ModelName: modelName, Weight: "1",
							}}},
						},
					},
				}},
			}},
			Recipes: []config.RoutingRecipe{
				{
					ID: recipeID, Revision: 1, Name: "private",
					Profile: config.RoutingProfile{
						Signals: config.Signals{
							MetadataRules: []config.MetadataRule{{
								Name: "tenant",
								Key:  "tenant",
								Predicate: config.MetadataPredicate{
									Equals: &expected,
								},
							}},
						},
						Decisions: []config.Decision{{
							ID: decisionID, Name: "private-route",
							Rules: config.RuleNode{
								Type: config.SignalTypeMetadata,
								Name: "tenant",
							},
						}},
					},
				},
			},
		}
		require.NoError(t, result.PrepareEntrypointRecipes())
		return result
	}

	initial := recipeConfig("alpha")
	classifiers, err := classification.BuildRecipeClassifiers(
		initial,
		nil,
		nil,
		nil,
	)
	require.NoError(t, err)
	service := NewRecipeClassificationService(classifiers, initial)

	require.NoError(t, service.TryRefreshRuntimeConfig(recipeConfig("beta")))
	response, err := service.ClassifyIntentForEval(IntentRequest{
		Text:     "hello",
		Model:    "router/private",
		Metadata: map[string]string{"tenant": "beta"},
	})
	require.NoError(t, err)
	assert.Equal(t, 1.0, response.SignalConfidences["metadata:tenant"])
	assert.Equal(t, config.RecipeName("private"), response.Recipe)
}
