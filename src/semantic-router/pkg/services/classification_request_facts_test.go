package services

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestClassificationServiceEvaluatesRequestEnvelopeFacts(t *testing.T) {
	equalsCanary := "canary"
	minBytes := 4.0
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
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
		},
	}
	classifier, err := classification.NewClassifier(cfg, nil, nil, nil)
	require.NoError(t, err)
	service := NewClassificationService(classifier, cfg)

	metadataResponse, err := service.ClassifyIntentForEval(IntentRequest{
		Metadata: map[string]string{"cohort": "canary"},
	})
	require.NoError(t, err)
	require.NotNil(t, metadataResponse.DecisionResult)
	require.Equal(t, "metadata-route", metadataResponse.DecisionResult.DecisionName)

	imageResponse, err := service.ClassifyIntentForEval(IntentRequest{
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
		Text: " \t \n",
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

func requestFactDecision(
	name string,
	signalType string,
	signalName string,
	priority int,
) config.Decision {
	return config.Decision{
		Name:     name,
		Priority: priority,
		Rules: config.RuleNode{
			Type: signalType,
			Name: signalName,
		},
	}
}

func TestClassifyIntentScopesSignalsToDefaultRecipeDecisions(t *testing.T) {
	equalsCanary := "canary"
	cfg := &config.RouterConfig{
		ExternalModels: []config.ExternalModelConfig{{
			Name:           "unrelated-judge",
			ModelRole:      config.ModelRoleClassification,
			ModelName:      "unrelated-judge",
			TimeoutSeconds: 1,
			ModelEndpoint: config.ClassifierVLLMEndpoint{
				Address:  "127.0.0.1",
				Port:     1,
				Protocol: "http",
			},
		}},
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				MetadataRules: []config.MetadataRule{{
					Name: "canary",
					Key:  "cohort",
					Predicate: config.MetadataPredicate{
						Equals: &equalsCanary,
					},
				}},
				ClassifierRules: []config.ClassifierSignalRule{{
					Name:         "other-recipe-risk",
					Type:         "llm",
					Model:        "unrelated-judge",
					Labels:       []string{"SAFE", "RISKY"},
					Instructions: "Classify.",
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
		},
	}
	classifier, err := classification.NewClassifier(cfg, nil, nil, nil)
	require.NoError(t, err)
	service := NewClassificationService(classifier, cfg)

	response, err := service.ClassifyIntent(IntentRequest{
		Metadata: map[string]string{"cohort": "canary"},
	})
	require.NoError(t, err)
	require.NotNil(t, response.DecisionResult)
	require.Equal(t, "metadata-route", response.DecisionResult.DecisionName)
	require.NotContains(t, response.SignalErrors, "classifier:other-recipe-risk")
}
