package extproc

import (
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func TestBuildModelSelectionConfigUsesDecisionScopedLearningState(t *testing.T) {
	got := buildModelSelectionConfig(learningStateRouterConfig())
	requireLearningConfigs(t, got)

	assertRLDrivenLearningConfig(t, got.RLDriven)
	assertGMTRouterLearningConfig(t, got.GMTRouter)
}

func learningStateRouterConfig() *config.RouterConfig {
	return &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				rlDrivenLearningDecision(),
				gmtRouterLearningDecision(),
			},
		},
	}
}

func rlDrivenLearningDecision() config.Decision {
	return config.Decision{
		Name: "rl-learning",
		Algorithm: &config.AlgorithmConfig{
			Type: string(selection.MethodRLDriven),
			RLDriven: &config.RLDrivenSelectionConfig{
				ExplorationRate:             0.15,
				ExplorationDecay:            0.97,
				MinExploration:              0.02,
				UseThompsonSampling:         true,
				EnablePersonalization:       true,
				PersonalizationBlend:        0.4,
				SessionContextWeight:        0.35,
				ImplicitFeedbackWeight:      0.45,
				CostAwareness:               true,
				CostWeight:                  0.25,
				StoragePath:                 "/var/lib/vsr/rl_state.json",
				AutoSaveInterval:            "45s",
				UseRouterR1Rewards:          true,
				CostRewardAlpha:             0.3,
				FormatRewardPenalty:         -0.75,
				EnableLLMRouting:            true,
				RouterR1ServerURL:           "http://router-r1:8080",
				LLMRoutingFallback:          "error",
				EnableMultiRoundAggregation: true,
				MaxAggregationRounds:        4,
			},
		},
	}
}

func gmtRouterLearningDecision() config.Decision {
	return config.Decision{
		Name: "gmt-learning",
		Algorithm: &config.AlgorithmConfig{
			Type: string(selection.MethodGMTRouter),
			GMTRouter: &config.GMTRouterSelectionConfig{
				EnablePersonalization:             true,
				HistorySampleSize:                 7,
				EmbeddingDimension:                1024,
				NumGNNLayers:                      3,
				AttentionHeads:                    12,
				MinInteractionsForPersonalization: 6,
				MaxInteractionsPerUser:            250,
				FeedbackTypes:                     []string{"rating", "ranking", "response"},
				ModelPath:                         "models/gmtrouter.pt",
				StoragePath:                       "/var/lib/vsr/gmt_graph.json",
			},
		},
	}
}

func TestBuildModelSelectionConfigPreservesLearningDefaultsWhenUnset(t *testing.T) {
	defaults := config.DefaultGlobalConfig()
	got := buildModelSelectionConfig(&defaults)

	defaultRouterDC := selection.DefaultRouterDCConfig()
	if !reflect.DeepEqual(got.RouterDC, defaultRouterDC) {
		t.Fatalf("RouterDC default drift:\n got: %+v\nwant: %+v", got.RouterDC, defaultRouterDC)
	}

	defaultHybrid := selection.DefaultHybridConfig()
	if !reflect.DeepEqual(got.Hybrid, defaultHybrid) {
		t.Fatalf("Hybrid default drift:\n got: %+v\nwant: %+v", got.Hybrid, defaultHybrid)
	}

	defaultRL := selection.DefaultRLDrivenConfig()
	if !reflect.DeepEqual(got.RLDriven, defaultRL) {
		t.Fatalf("RLDriven default drift:\n got: %+v\nwant: %+v", got.RLDriven, defaultRL)
	}

	defaultGMT := selection.DefaultGMTRouterConfig()
	if !reflect.DeepEqual(got.GMTRouter, defaultGMT) {
		t.Fatalf("GMTRouter default drift:\n got: %+v\nwant: %+v", got.GMTRouter, defaultGMT)
	}
}

func TestBuildModelSelectionConfigPreservesExplicitFalseOverrides(t *testing.T) {
	cfg := config.DefaultGlobalConfig()
	cfg.IntelligentRouting.ModelSelection.RouterDC.UseCapabilities = false
	cfg.IntelligentRouting.ModelSelection.Hybrid.NormalizeScores = false

	got := buildModelSelectionConfig(&cfg)

	if got.RouterDC == nil {
		t.Fatal("expected RouterDC config to be built")
	}
	if got.RouterDC.UseCapabilities {
		t.Fatalf("expected explicit router_dc.use_capabilities=false to be preserved, got %+v", got.RouterDC)
	}
	if got.RouterDC.Temperature != selection.DefaultRouterDCConfig().Temperature {
		t.Fatalf("expected router_dc defaults to survive explicit false override, got %+v", got.RouterDC)
	}

	if got.Hybrid == nil {
		t.Fatal("expected Hybrid config to be built")
	}
	if got.Hybrid.NormalizeScores {
		t.Fatalf("expected explicit hybrid.normalize_scores=false to be preserved, got %+v", got.Hybrid)
	}
	if got.Hybrid.RouterDCWeight != selection.DefaultHybridConfig().RouterDCWeight {
		t.Fatalf("expected hybrid defaults to survive explicit false override, got %+v", got.Hybrid)
	}
}

func TestBuildHybridSelectionConfigMergesDecisionOverrides(t *testing.T) {
	cfg := config.DefaultGlobalConfig()

	got := buildHybridSelectionConfig(&cfg, &config.HybridSelectionConfig{
		ExperienceWeight: 0.6,
		RouterDCWeight:   0.4,
	})

	if got == nil {
		t.Fatal("expected Hybrid config to be built")
	}
	if got.ExperienceWeight != 0.6 || got.RouterDCWeight != 0.4 {
		t.Fatalf("expected decision-scoped hybrid weights, got %+v", got)
	}
	if got.AutoMixWeight != selection.DefaultHybridConfig().AutoMixWeight || !got.NormalizeScores {
		t.Fatalf("expected decision-scoped hybrid config to preserve remaining defaults, got %+v", got)
	}
}

func TestBuildModelSelectionConfigUsesDecisionScopedMultiFactorConfig(t *testing.T) {
	got := buildModelSelectionConfig(&config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name: "weighted-latency-router",
					Algorithm: &config.AlgorithmConfig{
						Type: string(selection.MethodMultiFactor),
						MultiFactor: &config.MultiFactorSelectionConfig{
							Weights: &config.MultiFactorWeightsConfig{
								Quality: 0.7,
								Latency: 0.2,
								Cost:    0.1,
								Load:    0.0,
							},
							SLO: &config.MultiFactorSLOConfig{
								MaxTPOTMs:    85,
								MaxTTFTMs:    550,
								MaxCostPer1M: 2.4,
								MaxInflight:  32,
							},
							LatencyPercentile: 90,
							OnNoCandidates:    "fail",
						},
					},
				},
			},
		},
	})

	if got.MultiFactor == nil {
		t.Fatal("expected MultiFactor config to be built")
	}
	assertFloat(t, got.MultiFactor.Weights.Quality, 0.7, "multi_factor.weights.quality")
	assertFloat(t, got.MultiFactor.Weights.Latency, 0.2, "multi_factor.weights.latency")
	assertFloat(t, got.MultiFactor.Weights.Cost, 0.1, "multi_factor.weights.cost")
	assertFloat(t, got.MultiFactor.Weights.Load, 0.0, "multi_factor.weights.load")
	assertFloat(t, got.MultiFactor.SLO.MaxTPOTMs, 85, "multi_factor.slo.max_tpot_ms")
	assertFloat(t, got.MultiFactor.SLO.MaxTTFTMs, 550, "multi_factor.slo.max_ttft_ms")
	assertFloat(t, got.MultiFactor.SLO.MaxCostPer1M, 2.4, "multi_factor.slo.max_cost_per_1m")
	assertInt(t, got.MultiFactor.SLO.MaxInflight, 32, "multi_factor.slo.max_inflight")
	assertInt(t, got.MultiFactor.LatencyPercentile, 90, "multi_factor.latency_percentile")
	assertString(t, got.MultiFactor.OnNoCandidates, "fail", "multi_factor.on_no_candidates")
}

func assertFloat(t *testing.T, got, want float64, field string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s = %v, want %v", field, got, want)
	}
}

func requireLearningConfigs(t *testing.T, got *selection.ModelSelectionConfig) {
	t.Helper()
	if got.RLDriven == nil {
		t.Fatal("expected RLDriven config to be built")
	}
	if got.GMTRouter == nil {
		t.Fatal("expected GMTRouter config to be built")
	}
}

func assertRLDrivenLearningConfig(t *testing.T, got *selection.RLDrivenConfig) {
	t.Helper()
	assertFloat(t, got.ExplorationRate, 0.15, "rl_driven.exploration_rate")
	assertFloat(t, got.ExplorationDecay, 0.97, "rl_driven.exploration_decay")
	assertFloat(t, got.MinExploration, 0.02, "rl_driven.min_exploration")
	assertFloat(t, got.PersonalizationBlend, 0.4, "rl_driven.personalization_blend")
	assertFloat(t, got.SessionContextWeight, 0.35, "rl_driven.session_context_weight")
	assertFloat(t, got.ImplicitFeedbackWeight, 0.45, "rl_driven.implicit_feedback_weight")
	assertFloat(t, got.CostWeight, 0.25, "rl_driven.cost_weight")
	assertFloat(t, got.CostRewardAlpha, 0.3, "rl_driven.cost_reward_alpha")
	assertFloat(t, got.FormatRewardPenalty, -0.75, "rl_driven.format_reward_penalty")
	assertTrueFields(t, "rl_driven", map[string]bool{
		"use_thompson_sampling":          got.UseThompsonSampling,
		"enable_personalization":         got.EnablePersonalization,
		"cost_awareness":                 got.CostAwareness,
		"use_router_r1_rewards":          got.UseRouterR1Rewards,
		"enable_llm_routing":             got.EnableLLMRouting,
		"enable_multi_round_aggregation": got.EnableMultiRoundAggregation,
	})
	assertString(t, got.StoragePath, "/var/lib/vsr/rl_state.json", "rl_driven.storage_path")
	assertString(t, got.AutoSaveInterval, "45s", "rl_driven.auto_save_interval")
	assertString(t, got.RouterR1ServerURL, "http://router-r1:8080", "rl_driven.router_r1_server_url")
	assertString(t, got.LLMRoutingFallback, "error", "rl_driven.llm_routing_fallback")
	assertInt(t, got.MaxAggregationRounds, 4, "rl_driven.max_aggregation_rounds")
}

func assertGMTRouterLearningConfig(t *testing.T, got *selection.GMTRouterConfig) {
	t.Helper()
	assertTrueFields(t, "gmtrouter", map[string]bool{
		"enable_personalization": got.EnablePersonalization,
	})
	assertInt(t, got.HistorySampleSize, 7, "gmtrouter.history_sample_size")
	assertInt(t, got.EmbeddingDimension, 1024, "gmtrouter.embedding_dimension")
	assertInt(t, got.NumGNNLayers, 3, "gmtrouter.num_gnn_layers")
	assertInt(t, got.AttentionHeads, 12, "gmtrouter.attention_heads")
	assertInt(t, got.MinInteractionsForPersonalization, 6, "gmtrouter.min_interactions_for_personalization")
	assertInt(t, got.MaxInteractionsPerUser, 250, "gmtrouter.max_interactions_per_user")
	assertStringSlice(t, got.FeedbackTypes, []string{"rating", "ranking", "response"}, "gmtrouter.feedback_types")
	assertString(t, got.ModelPath, "models/gmtrouter.pt", "gmtrouter.model_path")
	assertString(t, got.StoragePath, "/var/lib/vsr/gmt_graph.json", "gmtrouter.storage_path")
}

func assertTrueFields(t *testing.T, family string, fields map[string]bool) {
	t.Helper()
	for field, got := range fields {
		if !got {
			t.Fatalf("%s.%s = false", family, field)
		}
	}
}

func assertString(t *testing.T, got, want, field string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s = %q, want %q", field, got, want)
	}
}

func assertInt(t *testing.T, got, want int, field string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s = %d, want %d", field, got, want)
	}
}

func assertStringSlice(t *testing.T, got, want []string, field string) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("%s = %v, want %v", field, got, want)
	}
}
