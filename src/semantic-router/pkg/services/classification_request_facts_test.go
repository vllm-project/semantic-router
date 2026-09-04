package services

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
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

func TestClassificationServiceContextSignalUsesFullRequestTokenFloor(t *testing.T) {
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
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
		},
	}
	classifier, err := classification.NewClassifier(cfg, nil, nil, nil)
	require.NoError(t, err)
	service := NewClassificationService(classifier, cfg)

	response, err := service.ClassifyIntentForEval(IntentRequest{
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
		Name:     name,
		Priority: priority,
		Rules: config.RuleNode{
			Type: signalType,
			Name: signalName,
		},
	}
}

func TestClassifyIntentForEvalReturnsDiagnosticsWhenFailRequest(t *testing.T) {
	threshold := 0.5
	cfg := &config.RouterConfig{
		ExternalModels: []config.ExternalModelConfig{{
			Name:           "risk-judge",
			ModelRole:      config.ModelRoleClassification,
			ModelName:      "risk-judge",
			TimeoutSeconds: 1,
			ModelEndpoint: config.ClassifierVLLMEndpoint{
				Address:  "127.0.0.1",
				Port:     1,
				Protocol: "http",
			},
		}},
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				ClassifierRules: []config.ClassifierSignalRule{{
					Name:         "risk",
					Type:         "llm",
					Model:        "risk-judge",
					Labels:       []string{"SAFE", "RISKY"},
					Instructions: "Classify.",
				}},
			},
			Decisions: []config.Decision{{
				Name: "guarded",
				Rules: config.RuleNode{
					Type:      config.SignalTypeClassifier,
					Name:      "risk",
					Label:     "RISKY",
					Predicate: &config.NumericPredicate{GTE: &threshold},
					OnUnknown: config.RuleOnUnknownFailRequest,
				},
			}},
		},
	}
	classifier, err := classification.NewClassifier(cfg, nil, nil, nil)
	require.NoError(t, err)
	service := NewClassificationService(classifier, cfg)

	for name, trace := range map[string]bool{"without trace": false, "with trace": true} {
		t.Run(name, func(t *testing.T) {
			response, evalErr := service.ClassifyIntentForEval(IntentRequest{
				Text:    "hello",
				Options: &IntentOptions{Trace: trace},
			})
			require.ErrorIs(t, evalErr, decision.ErrDecisionUnresolved)
			require.NotNil(t, response)
			require.NotEmpty(t, response.DecisionError)
			require.Equal(t, trace, len(response.EvalTrace) > 0)
		})
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
