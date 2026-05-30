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
	got := buildModelSelectionConfig(&config.RouterConfig{})

	defaultRL := selection.DefaultRLDrivenConfig()
	if !reflect.DeepEqual(got.RLDriven, defaultRL) {
		t.Fatalf("RLDriven default drift:\n got: %+v\nwant: %+v", got.RLDriven, defaultRL)
	}

	defaultGMT := selection.DefaultGMTRouterConfig()
	if !reflect.DeepEqual(got.GMTRouter, defaultGMT) {
		t.Fatalf("GMTRouter default drift:\n got: %+v\nwant: %+v", got.GMTRouter, defaultGMT)
	}
}

func TestBuildSessionAwareSelectionConfigPreservesExplicitZeroOverrides(t *testing.T) {
	baseIdleTimeout := 300
	baseSwitchMargin := 0.20
	baseStayBias := 0.20
	basePrefixCacheWeight := 0.30
	baseHandoffPenaltyWeight := 1.0
	baseDefaultHandoffPenalty := 0.10
	baseToolLoopStayBias := 0.40
	baseSwitchHistoryWeight := 0.20
	zeroInt := 0
	zeroFloat := 0.0

	routerCfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			ModelSelection: config.ModelSelectionConfig{
				SessionAware: config.SessionAwareSelectionConfig{
					IdleTimeoutSeconds:    &baseIdleTimeout,
					SwitchMargin:          &baseSwitchMargin,
					StayBias:              &baseStayBias,
					PrefixCacheWeight:     &basePrefixCacheWeight,
					HandoffPenaltyWeight:  &baseHandoffPenaltyWeight,
					DefaultHandoffPenalty: &baseDefaultHandoffPenalty,
					ToolLoopStayBias:      &baseToolLoopStayBias,
					SwitchHistoryWeight:   &baseSwitchHistoryWeight,
				},
			},
		},
	}

	got := buildSessionAwareSelectionConfig(routerCfg, &config.SessionAwareSelectionConfig{
		IdleTimeoutSeconds:    &zeroInt,
		MinTurnsBeforeSwitch:  &zeroInt,
		SwitchMargin:          &zeroFloat,
		StayBias:              &zeroFloat,
		PrefixCacheWeight:     &zeroFloat,
		HandoffPenaltyWeight:  &zeroFloat,
		DefaultHandoffPenalty: &zeroFloat,
		ToolLoopStayBias:      &zeroFloat,
		SwitchHistoryWeight:   &zeroFloat,
	})

	assertInt(t, got.IdleTimeoutSeconds, 0, "session_aware.idle_timeout_seconds")
	assertInt(t, got.MinTurnsBeforeSwitch, 0, "session_aware.min_turns_before_switch")
	assertFloat(t, got.SwitchMargin, 0, "session_aware.switch_margin")
	assertFloat(t, got.StayBias, 0, "session_aware.stay_bias")
	assertFloat(t, got.PrefixCacheWeight, 0, "session_aware.prefix_cache_weight")
	assertFloat(t, got.HandoffPenaltyWeight, 0, "session_aware.handoff_penalty_weight")
	assertFloat(t, got.DefaultHandoffPenalty, 0, "session_aware.default_handoff_penalty")
	assertFloat(t, got.ToolLoopStayBias, 0, "session_aware.tool_loop_stay_bias")
	assertFloat(t, got.SwitchHistoryWeight, 0, "session_aware.switch_history_weight")
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
