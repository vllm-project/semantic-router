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

	routerConfig := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{Name: "balanced_route"}},
		},
		Entrypoints: []config.EntrypointMapping{
			{ModelNames: []string{"amd/rocm-v1-flash"}, Recipe: speedRecipe},
		},
		Recipes: []config.RoutingRecipe{
			{Name: config.DefaultRecipeName, Profile: config.RoutingProfile{Decisions: []config.Decision{{Name: "balanced_route"}}}},
			{Name: speedRecipe, Profile: config.RoutingProfile{Decisions: []config.Decision{{Name: "flash_route"}}}},
		},
	}
	service := &ClassificationService{config: routerConfig}

	_, candidates, recipe, err := service.evalRoutingScope("amd/rocm-v1-flash")
	require.NoError(t, err)
	require.Len(t, candidates, 1)
	assert.Equal(t, "flash_route", candidates[0].Name)
	assert.Equal(t, speedRecipe, recipe)

	_, _, _, err = service.evalRoutingScope("amd/rocm-v1-missing")
	require.ErrorIs(t, err, ErrUnknownRoutingModel)

	response, err := service.ClassifyIntentForEval(IntentRequest{
		Text:  "hello",
		Model: "amd/rocm-v1-flash",
	})
	require.NoError(t, err)
	assert.Equal(t, speedRecipe, response.Recipe)

	_, err = service.ClassifyIntentForEval(IntentRequest{
		Text:  "hello",
		Model: "amd/rocm-v1-missing",
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

	classifier, err := service.classifierForRequestModel("")
	require.NoError(t, err)
	require.NotNil(t, classifier)
}

func TestRecipeClassificationServiceRefreshesNamedRecipePolicy(t *testing.T) {
	recipeConfig := func(expected string) *config.RouterConfig {
		return &config.RouterConfig{
			Entrypoints: []config.EntrypointMapping{{
				ModelNames: []string{"router/private"},
				Recipe:     "private",
			}},
			Recipes: []config.RoutingRecipe{
				{Name: config.DefaultRecipeName},
				{
					Name: "private",
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
							Name: "private-route",
							Rules: config.RuleNode{
								Type: config.SignalTypeMetadata,
								Name: "tenant",
							},
						}},
					},
				},
			},
		}
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
