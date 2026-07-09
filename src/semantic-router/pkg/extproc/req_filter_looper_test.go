package extproc

import (
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
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
		looperAlgorithms := []string{"confidence", "ratings", "fusion", "workflows"}

		for _, algorithmType := range looperAlgorithms {
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

func TestParseFusionRequestConfig(t *testing.T) {
	fusion, err := parseFusionRequestConfig([]byte(`{
		"model":"vllm-sr/fusion",
		"plugins":[{
			"id":"fusion",
			"model":"judge",
			"analysis_models":["panel-a","panel-b"],
			"max_concurrent":2,
			"max_completion_tokens":512,
			"temperature":0.2,
			"include_analysis":false,
			"include_intermediate_responses":false,
			"analysis_template":"analysis {{prompt}}",
			"synthesis_template":"synthesis {{analysis}}",
			"judge_prompt_version":"fusion-custom",
			"grounding":{
				"enabled":true,
				"reference":"context",
				"policy":"annotate",
				"min_score":0.4,
				"min_keep":2,
				"nli_contradiction_penalty":0.7,
				"on_error":"fail"
			}
		}]
	}`))
	require.NoError(t, err)
	require.NotNil(t, fusion)
	assert.Equal(t, "judge", fusion.Model)
	assert.Equal(t, []string{"panel-a", "panel-b"}, fusion.AnalysisModels)
	assert.Equal(t, 2, fusion.MaxConcurrent)
	assert.Equal(t, 512, fusion.MaxCompletionTokens)
	require.NotNil(t, fusion.Temperature)
	assert.Equal(t, 0.2, *fusion.Temperature)
	require.NotNil(t, fusion.IncludeAnalysis)
	assert.False(t, *fusion.IncludeAnalysis)
	require.NotNil(t, fusion.IncludeIntermediateResponses)
	assert.False(t, *fusion.IncludeIntermediateResponses)
	assert.Equal(t, "analysis {{prompt}}", fusion.AnalysisTemplate)
	assert.Equal(t, "synthesis {{analysis}}", fusion.SynthesisTemplate)
	assert.Equal(t, "fusion-custom", fusion.JudgePromptVersion)
	require.NotNil(t, fusion.Grounding)
	assert.True(t, fusion.Grounding.Enabled)
	assert.Equal(t, config.FusionGroundingReferenceContext, fusion.Grounding.Reference)
	assert.Equal(t, config.FusionGroundingPolicyAnnotate, fusion.Grounding.Policy)
	assert.Equal(t, 0.4, fusion.Grounding.MinScore)
	assert.Equal(t, 2, fusion.Grounding.MinKeep)
	assert.Equal(t, 0.7, fusion.Grounding.NLIContradictionPenalty)
	assert.Equal(t, config.FusionOnErrorFail, fusion.Grounding.OnError)
}

func TestParseFusionRequestConfigRejectsInvalidOnError(t *testing.T) {
	_, err := parseFusionRequestConfig([]byte(`{
		"model":"vllm-sr/fusion",
		"plugins":[{
			"id":"fusion",
			"on_error":"ignore"
		}]
	}`))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "on_error must be one of")
}

func TestResolveDirectFusionDecisionUsesMatchedFusionDecision(t *testing.T) {
	matchedDecision := &config.Decision{
		Name: "fusion-route",
		ModelRefs: []config.ModelRef{
			{Model: "panel-a"},
			{Model: "panel-b"},
		},
		Algorithm: &config.AlgorithmConfig{
			Type: "fusion",
			Fusion: &config.FusionAlgorithmConfig{
				Model: "judge",
			},
		},
	}
	router := &OpenAIRouter{
		Config: &config.RouterConfig{},
	}

	decision, status, err := router.resolveDirectFusionDecision(&RequestContext{
		VSRSelectedDecision: matchedDecision,
	})
	require.NoError(t, err)
	assert.Zero(t, status)
	require.NotNil(t, decision)
	assert.Equal(t, "fusion-route", decision.Name)
	assert.Equal(t, "fusion", decision.Algorithm.Type)
	assert.Equal(t, "judge", decision.Algorithm.Fusion.Model)
	require.Len(t, decision.ModelRefs, 2)
	assert.Equal(t, "panel-a", decision.ModelRefs[0].Model)
}

func TestResolveDirectFusionDecisionRejectsNonFusionMatchedDecision(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{},
	}

	_, status, err := router.resolveDirectFusionDecision(&RequestContext{
		VSRSelectedDecision: &config.Decision{
			Name:      "static-route",
			Algorithm: &config.AlgorithmConfig{Type: "static"},
		},
	})
	require.Error(t, err)
	assert.Equal(t, 400, status)
	assert.Contains(t, err.Error(), "no eligible Fusion decision matched")
}

func TestResolveDirectFusionDecisionAllowsRequestOnlyPluginPanel(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{
		OriginalRequestBody: []byte(`{
			"model":"vllm-sr/fusion",
			"plugins":[{
				"id":"fusion",
				"model":"judge",
				"analysis_models":["panel-a","panel-b"]
			}]
		}`),
	}

	decision, status, err := router.resolveDirectFusionDecision(ctx)
	require.NoError(t, err)
	assert.Zero(t, status)
	require.NotNil(t, decision)
	assert.Equal(t, directFusionDecisionName, decision.Name)
	assert.Equal(t, "fusion", decision.Algorithm.Type)
	assert.Empty(t, decision.ModelRefs)
}

func TestResolveDirectFusionDecisionFallsBackToConfiguredDecisionByPriority(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Decisions: []config.Decision{
					{
						Name:     "low-priority-fusion",
						Priority: 10,
						ModelRefs: []config.ModelRef{
							{Model: "panel-a"},
						},
						Algorithm: &config.AlgorithmConfig{Type: "fusion"},
					},
					{
						Name:     "high-priority-fusion",
						Priority: 20,
						ModelRefs: []config.ModelRef{
							{Model: "panel-b"},
						},
						Algorithm: &config.AlgorithmConfig{Type: "fusion"},
					},
				},
			},
		},
	}

	decision, status, err := router.resolveDirectFusionDecision(&RequestContext{
		RequestModel: "vllm-sr/fusion",
	})
	require.NoError(t, err)
	assert.Zero(t, status)
	require.NotNil(t, decision)
	assert.Equal(t, "high-priority-fusion", decision.Name)
}

func TestDecisionCandidatesForFusionModelOnlyIncludesFusionDecisions(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Decisions: []config.Decision{
					{
						Name:      "static-route",
						Algorithm: &config.AlgorithmConfig{Type: "static"},
					},
					{
						Name:      "fusion-route",
						Algorithm: &config.AlgorithmConfig{Type: "fusion"},
					},
					{
						Name:      "remom-route",
						Algorithm: &config.AlgorithmConfig{Type: "remom"},
					},
					{
						Name:      "flow-route",
						Algorithm: &config.AlgorithmConfig{Type: "workflows"},
					},
				},
			},
		},
	}

	assert.Nil(t, router.decisionCandidatesForRequestModel("vllm-sr/auto"))

	candidates := router.decisionCandidatesForRequestModel("vllm-sr/fusion")
	require.Len(t, candidates, 1)
	assert.Equal(t, "fusion-route", candidates[0].Name)
}

func TestDecisionCandidatesForReMoMModelOnlyIncludesReMoMDecisions(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Decisions: []config.Decision{
					{
						Name:      "static-route",
						Algorithm: &config.AlgorithmConfig{Type: "static"},
					},
					{
						Name:      "fusion-route",
						Algorithm: &config.AlgorithmConfig{Type: "fusion"},
					},
					{
						Name:      "remom-route",
						Algorithm: &config.AlgorithmConfig{Type: "remom"},
					},
					{
						Name:      "flow-route",
						Algorithm: &config.AlgorithmConfig{Type: "workflows"},
					},
				},
			},
		},
	}

	candidates := router.decisionCandidatesForRequestModel("vllm-sr/remom")
	require.Len(t, candidates, 1)
	assert.Equal(t, "remom-route", candidates[0].Name)
}

func TestDecisionCandidatesForFlowModelOnlyIncludesWorkflowDecisions(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Decisions: []config.Decision{
					{
						Name:      "static-route",
						Algorithm: &config.AlgorithmConfig{Type: "static"},
					},
					{
						Name:      "fusion-route",
						Algorithm: &config.AlgorithmConfig{Type: "fusion"},
					},
					{
						Name:      "remom-route",
						Algorithm: &config.AlgorithmConfig{Type: "remom"},
					},
					{
						Name:      "flow-route",
						Algorithm: &config.AlgorithmConfig{Type: "workflows"},
					},
				},
			},
		},
	}

	candidates := router.decisionCandidatesForRequestModel("vllm-sr/flow")
	require.Len(t, candidates, 1)
	assert.Equal(t, "flow-route", candidates[0].Name)
}

func TestResolveDirectReMoMDecisionRejectsNonReMoMMatchedDecision(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}

	_, status, err := router.resolveDirectReMoMDecision(&RequestContext{
		RequestModel: "vllm-sr/remom",
		VSRSelectedDecision: &config.Decision{
			Name:      "static-route",
			Algorithm: &config.AlgorithmConfig{Type: "static"},
		},
	})
	require.Error(t, err)
	assert.Equal(t, 400, status)
	assert.Contains(t, err.Error(), "no eligible ReMoM decision matched")
}

func TestResolveDirectReMoMDecisionFallsBackToHighestPriorityReMoMDecision(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Decisions: []config.Decision{
					{
						Name:     "low-priority-remom",
						Priority: 10,
						ModelRefs: []config.ModelRef{
							{Model: "panel-a"},
						},
						Algorithm: &config.AlgorithmConfig{
							Type:  "remom",
							ReMoM: &config.ReMoMAlgorithmConfig{BreadthSchedule: []int{1}},
						},
					},
					{
						Name:     "high-priority-remom",
						Priority: 20,
						ModelRefs: []config.ModelRef{
							{Model: "panel-b"},
						},
						Algorithm: &config.AlgorithmConfig{
							Type:  "remom",
							ReMoM: &config.ReMoMAlgorithmConfig{BreadthSchedule: []int{1}},
						},
					},
				},
			},
		},
	}

	decision, status, err := router.resolveDirectReMoMDecision(&RequestContext{
		RequestModel: "vllm-sr/remom",
	})
	require.NoError(t, err)
	assert.Zero(t, status)
	require.NotNil(t, decision)
	assert.Equal(t, "high-priority-remom", decision.Name)
}

func TestResolveDirectFlowDecisionRejectsNonFlowMatchedDecision(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}

	_, status, err := router.resolveDirectFlowDecision(&RequestContext{
		RequestModel: "vllm-sr/flow",
		VSRSelectedDecision: &config.Decision{
			Name:      "static-route",
			Algorithm: &config.AlgorithmConfig{Type: "static"},
		},
	})
	require.Error(t, err)
	assert.Equal(t, 400, status)
	assert.Contains(t, err.Error(), "no eligible Flow decision matched")
}

func TestResolveDirectFlowDecisionFallsBackToConfiguredDecisionByPriority(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Decisions: []config.Decision{
					{
						Name:     "low-priority-flow",
						Priority: 10,
						ModelRefs: []config.ModelRef{
							{Model: "worker-a"},
						},
						Algorithm: &config.AlgorithmConfig{Type: "workflows"},
					},
					{
						Name:     "high-priority-flow",
						Priority: 20,
						ModelRefs: []config.ModelRef{
							{Model: "worker-b"},
						},
						Algorithm: &config.AlgorithmConfig{Type: "workflows"},
					},
				},
			},
		},
	}

	decision, status, err := router.resolveDirectFlowDecision(&RequestContext{
		RequestModel: "vllm-sr/flow",
	})
	require.NoError(t, err)
	assert.Zero(t, status)
	require.NotNil(t, decision)
	assert.Equal(t, "high-priority-flow", decision.Name)
}

func TestCreateLooperResponseIncludesTrackedHeaders(t *testing.T) {
	resp := &looper.Response{
		Body:          []byte(`{"ok":true}`),
		ContentType:   "application/json",
		Model:         "model-b",
		ModelsUsed:    []string{"model-a", "model-b"},
		Iterations:    2,
		AlgorithmType: "elo",
	}
	// The looper trace, matched signals, category and session phase are demoted
	// to x-vsr-debug (#2205), so this tracked-headers test opts into debug.
	reqCtx := &RequestContext{
		Headers:                       map[string]string{headers.VSRDebug: "true"},
		VSRMatchedKeywords:            []string{"python"},
		VSRMatchedEmbeddings:          []string{"coding"},
		VSRMatchedContext:             []string{"memory"},
		VSRMatchedComplexity:          []string{"complexity:medium"},
		VSRMatchedModality:            []string{"AR"},
		VSRMatchedAuthz:               []string{"authz:team-a"},
		VSRMatchedJailbreak:           []string{"jailbreak:block"},
		VSRMatchedPII:                 []string{"pii:email"},
		VSRMatchedReask:               []string{"likely_dissatisfied"},
		VSRMatchedProjection:          []string{"balance_reasoning"},
		VSRContextTokenCount:          42,
		VSRSelectedModel:              "model-b",
		VSRSelectedDecisionName:       "coding",
		VSRSelectedDecisionConfidence: 0,
		VSRSelectedCategory:           "programming",
		RouterReplayID:                "replay-123",
		VSRLearningPolicies: testLearningPolicies(
			replayTestProtectionPolicyWithTrace(&selection.SessionPolicyTrace{
				Phase: "tool_loop",
			}),
		),
	}

	response := (&OpenAIRouter{}).createLooperResponse(resp, reqCtx)
	headerMap := headerValuesByName(response.GetImmediateResponse().Headers.SetHeaders)

	assert.Equal(t, "application/json", headerMap["content-type"])
	assert.Equal(t, "model-b", headerMap[headers.VSRLooperModel])
	assert.Equal(t, "model-a,model-b", headerMap[headers.VSRLooperModelsUsed])
	assert.Equal(t, "2", headerMap[headers.VSRLooperIterations])
	assert.Equal(t, "elo", headerMap[headers.VSRLooperAlgorithm])
	assert.Equal(t, "python", headerMap[headers.VSRMatchedKeywords])
	assert.Equal(t, "complexity:medium", headerMap[headers.VSRMatchedComplexity])
	assert.Equal(t, "AR", headerMap[headers.VSRMatchedModality])
	assert.Equal(t, "authz:team-a", headerMap[headers.VSRMatchedAuthz])
	assert.Equal(t, "jailbreak:block", headerMap[headers.VSRMatchedJailbreak])
	assert.Equal(t, "pii:email", headerMap[headers.VSRMatchedPII])
	assert.Equal(t, "likely_dissatisfied", headerMap[headers.VSRMatchedReask])
	assert.Equal(t, "balance_reasoning", headerMap[headers.VSRMatchedProjection])
	assert.Equal(t, "model-b", headerMap[headers.VSRSelectedModel])
	assert.Equal(t, "coding", headerMap[headers.VSRSelectedDecision])
	assert.Equal(t, "0.0000", headerMap[headers.VSRSelectedConfidence])
	assert.Equal(t, "programming", headerMap[headers.VSRSelectedCategory])
	assert.Equal(t, "tool_loop", headerMap[headers.VSRSessionPhase])
	assert.Equal(t, "replay-123", headerMap[headers.RouterReplayID])
	assert.Equal(t, "42", headerMap[headers.VSRContextTokenCount])
}

func TestCreateLooperResponseDefaultSurfaceIsLean(t *testing.T) {
	// Without x-vsr-debug the looper response carries only content-type, the
	// keystone headers and the final routing facts (#2205). The execution trace,
	// matched signals, category and session phase are demoted.
	resp := &looper.Response{
		Body:          []byte(`{"ok":true}`),
		ContentType:   "application/json",
		Model:         "model-b",
		ModelsUsed:    []string{"model-a", "model-b"},
		Iterations:    2,
		AlgorithmType: "elo",
	}
	reqCtx := &RequestContext{
		VSRMatchedKeywords:            []string{"python"},
		VSRContextTokenCount:          42,
		VSRSelectedModel:              "model-b",
		VSRSelectedDecisionName:       "coding",
		VSRSelectedDecisionConfidence: 0,
		VSRSelectedCategory:           "programming",
		RouterReplayID:                "replay-123",
		VSRLearningPolicies: testLearningPolicies(
			replayTestProtectionPolicyWithTrace(&selection.SessionPolicyTrace{
				Phase: "tool_loop",
			}),
		),
	}

	response := (&OpenAIRouter{}).createLooperResponse(resp, reqCtx)
	headerMap := headerValuesByName(response.GetImmediateResponse().Headers.SetHeaders)

	// content-type, keystone and final routing facts ride on the default surface.
	assert.Equal(t, "application/json", headerMap["content-type"])
	assert.Equal(t, headers.SchemaVersionValue, headerMap[headers.VSRSchemaVersion])
	assert.Equal(t, headers.ResponsePathLooper, headerMap[headers.VSRResponsePath])
	assert.Equal(t, "model-b", headerMap[headers.VSRSelectedModel])
	assert.Equal(t, "coding", headerMap[headers.VSRSelectedDecision])
	assert.Equal(t, "0.0000", headerMap[headers.VSRSelectedConfidence])
	assert.Equal(t, "replay-123", headerMap[headers.RouterReplayID])

	// The demoted detail headers are absent.
	assert.NotContains(t, headerMap, headers.VSRLooperModel)
	assert.NotContains(t, headerMap, headers.VSRLooperIterations)
	assert.NotContains(t, headerMap, headers.VSRMatchedKeywords)
	assert.NotContains(t, headerMap, headers.VSRContextTokenCount)
	assert.NotContains(t, headerMap, headers.VSRSelectedCategory)
	assert.NotContains(t, headerMap, headers.VSRSessionPhase)
}

func TestGetReasoningInfoFromDecision(t *testing.T) {
	useReasoning := true
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				ReasoningConfig: config.ReasoningConfig{DefaultReasoningEffort: "high"},
			},
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"model-a": {ReasoningFamily: "qwen3"},
					"model-b": {ReasoningFamily: "deepseek"},
				},
			},
		},
	}
	decision := &config.Decision{
		ModelRefs: []config.ModelRef{
			{
				Model: "model-a",
				ModelReasoningControl: config.ModelReasoningControl{
					UseReasoning:    &useReasoning,
					ReasoningEffort: "low",
				},
			},
		},
	}

	use, effort := router.getReasoningInfoFromDecision(decision, "model-a")
	assert.True(t, use)
	assert.Equal(t, "low", effort)

	use, effort = router.getReasoningInfoFromDecision(decision, "model-b")
	assert.True(t, use)
	assert.Equal(t, "high", effort)
}

func TestBuildHeaderMutationsForLooperIncludesAuthorizationAndPluginHeaders(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"model-a": {AccessKey: "secret"},
				},
			},
		},
	}
	decision := &config.Decision{
		Name: "coding",
		Plugins: []config.DecisionPlugin{
			{
				Type: "header_mutation",
				Configuration: config.MustStructuredPayload(map[string]interface{}{
					"add": []map[string]interface{}{
						{"name": "x-extra", "value": "1"},
					},
					"delete": []string{"x-remove-me"},
				}),
			},
		},
	}

	setHeaders, removeHeaders, errorResponse := router.buildHeaderMutationsForLooper(
		decision,
		"model-a",
		looperBackendRoute{},
		&RequestContext{},
	)
	require.Nil(t, errorResponse)
	headerMap := headerValuesByName(setHeaders)

	assert.Equal(t, "model-a", headerMap[headers.SelectedModel])
	assert.Equal(t, "model-a", headerMap[headers.VSRSelectedModel])
	assert.Equal(t, "Bearer secret", headerMap["Authorization"])
	assert.Equal(t, "1", headerMap["x-extra"])
	assert.Contains(t, removeHeaders, "content-length")
	assert.Contains(t, removeHeaders, "x-remove-me")
}

func TestBuildHeaderMutationsForLooperUsesProviderCredentialResolver(t *testing.T) {
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"panel-a": {
					PreferredEndpoints: []string{"openrouter"},
					AccessKey:          "secret",
				},
			},
			VLLMEndpoints: []config.VLLMEndpoint{
				{
					Name:                "openrouter",
					ProviderProfileName: "openrouter",
					Weight:              1,
				},
			},
			ProviderProfiles: map[string]config.ProviderProfile{
				"openrouter": {
					Type:       "openai",
					BaseURL:    "https://openrouter.ai/api/v1",
					AuthHeader: "Authorization",
					AuthPrefix: "Bearer",
				},
			},
		},
	}
	router := &OpenAIRouter{
		Config:             cfg,
		CredentialResolver: newTestCredentialResolver(cfg),
	}
	ctx := &RequestContext{Headers: map[string]string{}}
	route, err := router.resolveLooperBackendRoute(ctx, "panel-a")
	require.NoError(t, err)

	setHeaders, removeHeaders, errorResponse := router.buildHeaderMutationsForLooper(nil, "panel-a", route, ctx)
	require.Nil(t, errorResponse)
	headerMap := headerValuesByName(setHeaders)

	assert.Equal(t, "panel-a", headerMap[headers.SelectedModel])
	assert.Equal(t, "panel-a", headerMap[headers.VSRSelectedModel])
	assert.Equal(t, "Bearer secret", headerMap["Authorization"])
	assert.Equal(t, "/api/v1/chat/completions", headerMap[":path"])
	assert.Contains(t, removeHeaders, "content-length")
}

func TestHandleLooperInternalRequestRewritesModel(t *testing.T) {
	ctx := &RequestContext{
		OriginalRequestBody: []byte(`{"model":"auto","messages":[{"role":"user","content":"hi"}]}`),
	}

	response, err := (&OpenAIRouter{}).handleLooperInternalRequest("model-b", ctx)
	require.NoError(t, err)
	require.NotNil(t, response.GetRequestBody())

	body := response.GetRequestBody().Response.GetBodyMutation().GetBody()
	assert.JSONEq(
		t,
		`{"model":"model-b","messages":[{"role":"user","content":"hi"}]}`,
		string(body),
	)

	headerMap := headerValuesByName(response.GetRequestBody().Response.HeaderMutation.SetHeaders)
	assert.Equal(t, "model-b", headerMap[headers.SelectedModel])
	assert.Equal(t, "model-b", headerMap[headers.VSRSelectedModel])
	assert.Contains(t, response.GetRequestBody().Response.HeaderMutation.RemoveHeaders, "content-length")
}

func TestHandleLooperInternalRequestWithPluginsResolvesProviderModelAlias(t *testing.T) {
	router := &OpenAIRouter{
		Cache: &spyCache{},
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Decisions: []config.Decision{
					{
						Name:      "fusion_alias",
						ModelRefs: []config.ModelRef{{Model: "panel-a"}},
					},
				},
			},
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"panel-a": {
						PreferredEndpoints: []string{"panel-backend"},
						ExternalModelIDs: map[string]string{
							"vllm": "qwen2.5-omni-7b",
						},
					},
				},
				VLLMEndpoints: []config.VLLMEndpoint{
					{
						Name:    "panel-backend",
						Address: "127.0.0.1",
						Port:    8000,
						Type:    "vllm",
						Weight:  1,
					},
				},
			},
		},
	}
	router.CredentialResolver = newTestCredentialResolver(router.Config)
	ctx := &RequestContext{
		LooperRequest: true,
		Headers: map[string]string{
			headers.VSRLooperDecision: "fusion_alias",
		},
		OriginalRequestBody: []byte(`{"model":"panel-a","messages":[{"role":"user","content":"hi"}]}`),
	}

	response, err := router.handleLooperInternalRequestWithPlugins("panel-a", ctx)
	require.NoError(t, err)
	require.NotNil(t, response.GetRequestBody())

	body := response.GetRequestBody().Response.GetBodyMutation().GetBody()
	assert.Contains(t, string(body), `"model":"qwen2.5-omni-7b"`)

	headerMap := headerValuesByName(response.GetRequestBody().Response.HeaderMutation.SetHeaders)
	assert.Equal(t, "panel-a", headerMap[headers.SelectedModel])
	assert.Equal(t, "panel-a", headerMap[headers.VSRSelectedModel])
}

func headerValuesByName(headers []*core.HeaderValueOption) map[string]string {
	result := make(map[string]string, len(headers))
	for _, header := range headers {
		result[header.Header.Key] = string(header.Header.RawValue)
	}
	return result
}
