package services

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestClassificationServiceEvaluatesRequestEnvelopeFacts(t *testing.T) {
	equalsCanary := "canary"
	minBytes := 4.0
	service, requestModel := newRequestFactsClassificationService(t, config.RoutingProfile{
		Signals: config.Signals{
			MetadataRules: []config.MetadataRule{{
				Name: "canary",
				Key:  "cohort",
				Predicate: config.MetadataPredicate{
					Equals: &equalsCanary,
				},
			}},
			ConversationRules: []config.ConversationRule{{
				Name: "has-image",
				Feature: config.ConversationFeature{
					Type:   "exists",
					Source: config.ConversationSource{Type: "image_content"},
				},
			}},
			StructureRules: []config.StructureRule{{
				Name: "raw-bytes",
				Feature: config.StructureFeature{
					Type:   "count",
					Source: config.StructureSource{Type: "text_bytes"},
				},
				Predicate: &config.NumericPredicate{GTE: &minBytes},
			}},
		},
		Decisions: []config.Decision{
			requestFactDecision("metadata-route", config.SignalTypeMetadata, "canary", 30),
			requestFactDecision("image-route", config.SignalTypeConversation, "has-image", 20),
			requestFactDecision("bytes-route", config.SignalTypeStructure, "raw-bytes", 10),
		},
	}, nil)

	metadataResponse, err := service.ClassifyIntentForEval(IntentRequest{
		Metadata: map[string]string{"cohort": "canary"},
		Model:    requestModel,
	})
	require.NoError(t, err)
	require.NotNil(t, metadataResponse.DecisionResult)
	require.Equal(t, "metadata-route", metadataResponse.DecisionResult.DecisionName)

	imageResponse, err := service.ClassifyIntentForEval(IntentRequest{
		Model: requestModel,
		Messages: []IntentMessage{{
			Role: "user",
			Content: mustMessageContent(t, []map[string]interface{}{{
				"type":      "image_url",
				"image_url": map[string]string{"url": "https://example.invalid/image.png"},
			}}),
		}},
	})
	require.NoError(t, err)
	require.NotNil(t, imageResponse.DecisionResult)
	require.Equal(t, "image-route", imageResponse.DecisionResult.DecisionName)
	require.Equal(t, float64(1), imageResponse.SignalValues["conversation:has-image"])

	bytesResponse, err := service.ClassifyIntentForEval(IntentRequest{
		Model: requestModel,
		Messages: []IntentMessage{{
			Role:    "user",
			Content: mustMessageContent(t, " \t \n"),
		}},
	})
	require.NoError(t, err)
	require.NotNil(t, bytesResponse.DecisionResult)
	require.Equal(t, "bytes-route", bytesResponse.DecisionResult.DecisionName)
	require.Equal(t, float64(4), bytesResponse.SignalValues["structure:raw-bytes"])

	topLevelBytesResponse, err := service.ClassifyIntentForEval(IntentRequest{
		Text:  " \t \n",
		Model: requestModel,
	})
	require.NoError(t, err)
	require.NotNil(t, topLevelBytesResponse.DecisionResult)
	require.Equal(
		t,
		"bytes-route",
		topLevelBytesResponse.DecisionResult.DecisionName,
	)
	require.Equal(
		t,
		float64(4),
		topLevelBytesResponse.SignalValues["structure:raw-bytes"],
	)
}

func TestClassificationServiceContextSignalUsesFullRequestTokenFloor(t *testing.T) {
	service, requestModel := newRequestFactsClassificationService(t, config.RoutingProfile{
		Signals: config.Signals{
			ContextRules: []config.ContextRule{
				{
					Name:      "short-request-context",
					MinTokens: config.TokenCount("0"),
					MaxTokens: config.TokenCount("10K"),
				},
				{
					Name:      "large-request-context",
					MinTokens: config.TokenCount("10K"),
					MaxTokens: config.TokenCount("128K"),
				},
			},
		},
		Decisions: []config.Decision{
			requestFactDecision(
				"large-context-route",
				config.SignalTypeContext,
				"large-request-context",
				10,
			),
		},
	}, nil)

	response, err := service.ClassifyIntentForEval(IntentRequest{
		Model: requestModel,
		Messages: []IntentMessage{
			{Role: "user", Content: mustMessageContent(t, strings.Repeat("p", 8_000))},
			{
				Role:       "tool",
				ToolCallID: "call-1",
				Content: mustMessageContent(
					t,
					strings.Repeat(`{"row":9007199254740993123456789}`, 100),
				),
			},
			{
				Role: "user",
				Content: mustMessageContent(t, []map[string]any{
					{"type": "text", "text": "ok"},
					{
						"type": "image_url",
						"image_url": map[string]string{
							"url": "data:image/png;base64,PRIVATE",
						},
					},
				}),
			},
		},
		Tools: []json.RawMessage{mustMessageContent(t, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        "lookup",
				"description": strings.Repeat("schema", 500),
				"parameters":  map[string]any{"type": "object"},
			},
		})},
		MaxCompletionTokens: json.RawMessage(`4096`),
	})
	require.NoError(t, err)
	require.NotNil(t, response.DecisionResult)
	require.Equal(t, "large-context-route", response.DecisionResult.DecisionName)
	require.NotNil(t, response.DecisionResult.MatchedSignals)
	require.Contains(t, response.DecisionResult.MatchedSignals.Context, "large-request-context")
}

func requestFactDecision(
	name string,
	signalType string,
	signalName string,
	priority int,
) config.Decision {
	return config.Decision{
		ID:       "decision-" + name,
		Name:     name,
		Priority: priority,
		Rules: config.RuleNode{
			Type: signalType,
			Name: signalName,
		},
	}
}

func TestClassifyIntentScopesSignalsToSelectedEntrypointRecipe(t *testing.T) {
	equalsCanary := "canary"
	service, requestModel := newRequestFactsClassificationService(t, config.RoutingProfile{
		Signals: config.Signals{
			MetadataRules: []config.MetadataRule{{
				Name: "canary",
				Key:  "cohort",
				Predicate: config.MetadataPredicate{
					Equals: &equalsCanary,
				},
			}},
		},
		Decisions: []config.Decision{
			requestFactDecision(
				"metadata-route",
				config.SignalTypeMetadata,
				"canary",
				10,
			),
		},
	}, func(cfg *config.RouterConfig) {
		cfg.ExternalModels = []config.ExternalModelConfig{{
			Name:           "unrelated-judge",
			ModelRole:      config.ModelRoleClassification,
			ModelName:      "unrelated-judge",
			TimeoutSeconds: 1,
			ModelEndpoint: config.ClassifierVLLMEndpoint{
				Address:  "127.0.0.1",
				Port:     1,
				Protocol: "http",
			},
		}}
		cfg.Recipes = append(cfg.Recipes, config.RoutingRecipe{
			ID: "recipe-unrelated", Revision: 1, Name: "unrelated",
			Profile: config.RoutingProfile{
				Signals: config.Signals{ClassifierRules: []config.ClassifierSignalRule{{
					Name:         "other-recipe-risk",
					Type:         "llm",
					Model:        "unrelated-judge",
					Labels:       []string{"SAFE", "RISKY"},
					Instructions: "Classify.",
				}}},
				Decisions: []config.Decision{requestFactDecision(
					"unrelated-route",
					config.SignalTypeClassifier,
					"other-recipe-risk",
					10,
				)},
			},
		})
	})

	response, err := service.ClassifyIntent(IntentRequest{
		Metadata: map[string]string{"cohort": "canary"},
		Model:    requestModel,
	})
	require.NoError(t, err)
	require.NotNil(t, response.DecisionResult)
	require.Equal(t, "metadata-route", response.DecisionResult.DecisionName)
	require.NotContains(t, response.SignalErrors, "classifier:other-recipe-risk")
}

func newRequestFactsClassificationService(
	t *testing.T,
	profile config.RoutingProfile,
	configure func(*config.RouterConfig),
) (*ClassificationService, string) {
	t.Helper()
	const (
		requestModel = "router/request-facts"
		backendModel = "backend/request-facts"
		backendID    = "model-request-facts"
		recipeID     = "recipe-request-facts"
	)
	assignments := make(map[string]config.RoutingAssignmentSet, len(profile.Decisions))
	for index := range profile.Decisions {
		decision := &profile.Decisions[index]
		if decision.ID == "" {
			decision.ID = "decision-" + decision.Name
		}
		assignments[decision.ID] = config.RoutingAssignmentSet{Models: []config.RoutingModelAssignment{{
			ModelID: backendID, ModelRevision: 1, ModelName: backendModel, Weight: "1",
		}}}
	}
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			backendModel: {ResourceID: backendID, ResourceRevision: 1},
		}},
		Recipes: []config.RoutingRecipe{{
			ID: recipeID, Revision: 1, Name: "request-facts", Profile: profile,
		}},
		Entrypoints: []config.EntrypointMapping{{
			ID: "entrypoint-request-facts", Revision: 1, Name: requestModel, ModelNames: []string{requestModel},
			Rules: []config.EntrypointRule{{
				ID: "rule-request-facts", Name: "default",
				Action: config.EntrypointRuleAction{
					RecipeID: recipeID, RecipeRevision: 1, Recipe: "request-facts", Assignments: assignments,
				},
			}},
		}},
	}
	if configure != nil {
		configure(cfg)
	}
	require.NoError(t, cfg.PrepareEntrypointRecipes())
	classifiers, err := classification.BuildRecipeClassifiers(cfg, nil, nil, nil)
	require.NoError(t, err)
	return NewRecipeClassificationService(classifiers, cfg), requestModel
}
