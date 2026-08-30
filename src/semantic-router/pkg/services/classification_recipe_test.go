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
			{ModelNames: []string{"router/speed-flash"}, Recipe: speedRecipe},
		},
		Recipes: []config.RoutingRecipe{
			{Name: config.DefaultRecipeName, Profile: config.RoutingProfile{Decisions: []config.Decision{{Name: "balanced_route"}}}},
			{Name: speedRecipe, Profile: config.RoutingProfile{Decisions: []config.Decision{{Name: "flash_route"}}}},
		},
	}
	service := &ClassificationService{config: routerConfig}

	_, candidates, recipe, err := service.evalRoutingScope("router/speed-flash")
	require.NoError(t, err)
	require.Len(t, candidates, 1)
	assert.Equal(t, "flash_route", candidates[0].Name)
	assert.Equal(t, speedRecipe, recipe)

	_, _, _, err = service.evalRoutingScope("router/missing")
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

// TestConditionalEntrypointUnresolvableFromClassificationAPI documents a
// real, intentional Phase-1 limitation of conditional (rules-based)
// entrypoints (issue #2868): the classification API has no HTTP request —
// no headers, no path — to evaluate a caller's rules against, so a virtual
// model name behind conditional rules must be rejected here rather than
// guessed. Before RecipeForRequestModel/RecipeForRoutingModel were made
// Rules-aware, this call site would have silently treated the name as an
// unclaimed/unknown model too, but for the wrong reason (it never even
// checked whether the name was a real entrypoint); now it's an explicit,
// documented rejection. Extending eval with a request_context (per the
// issue's own proposed design) is the natural way to lift this limitation
// later — deliberately out of scope here.
func TestConditionalEntrypointUnresolvableFromClassificationAPI(t *testing.T) {
	routerConfig := &config.RouterConfig{
		Entrypoints: []config.EntrypointMapping{{
			ModelNames: []string{"router/tenant-auto"},
			Rules: []config.EntrypointRule{{
				Name:   "tenant-a",
				Recipe: config.DefaultRecipeName,
				Matches: []config.EntrypointMatch{{
					Headers: []config.HeaderMatcher{{Name: "x-authz-tenant-id", Type: config.HeaderMatchExact, Value: "A"}},
				}},
			}},
		}},
		Recipes: []config.RoutingRecipe{{Name: config.DefaultRecipeName}},
	}

	// IsEntrypointModelName must still recognize the name as claimed (it's a
	// pure alias-membership test, unaffected by the missing request
	// context) — only recipe *resolution* is context-free-unresolvable.
	require.True(t, routerConfig.IsEntrypointModelName("router/tenant-auto"))

	classifiers, err := classification.BuildRecipeClassifiers(routerConfig, nil, nil, nil)
	require.NoError(t, err)
	service := NewRecipeClassificationService(classifiers, routerConfig)

	_, _, _, err = service.evalRoutingScope("router/tenant-auto")
	require.ErrorIs(t, err, ErrUnknownRoutingModel)

	_, err = service.classifierForRequestModel("router/tenant-auto")
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
