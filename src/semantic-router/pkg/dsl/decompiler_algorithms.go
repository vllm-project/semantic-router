package dsl

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

type algorithmFieldExporter func(*config.AlgorithmConfig, map[string]Value)

var algorithmFieldExporters = map[string]algorithmFieldExporter{
	"confidence": func(algo *config.AlgorithmConfig, fields map[string]Value) {
		confidenceAlgorithmToFields(algo.Confidence, fields)
	},
	"ratings": func(algo *config.AlgorithmConfig, fields map[string]Value) {
		ratingsAlgorithmToFields(algo.Ratings, fields)
	},
	"remom": func(algo *config.AlgorithmConfig, fields map[string]Value) {
		remomAlgorithmToFields(algo.ReMoM, fields)
	},
	"elo": func(algo *config.AlgorithmConfig, fields map[string]Value) {
		eloAlgorithmToFields(algo.Elo, fields)
	},
	"router_dc": func(algo *config.AlgorithmConfig, fields map[string]Value) {
		routerDCAlgorithmToFields(algo.RouterDC, fields)
	},
	"automix": func(algo *config.AlgorithmConfig, fields map[string]Value) {
		autoMixAlgorithmToFields(algo.AutoMix, fields)
	},
	"hybrid": func(algo *config.AlgorithmConfig, fields map[string]Value) {
		hybridAlgorithmToFields(algo.Hybrid, fields)
	},
	"rl_driven": func(algo *config.AlgorithmConfig, fields map[string]Value) {
		rlDrivenAlgorithmToFields(algo.RLDriven, fields)
	},
	"gmtrouter": func(algo *config.AlgorithmConfig, fields map[string]Value) {
		gmtRouterAlgorithmToFields(algo.GMTRouter, fields)
	},
	"latency_aware": func(algo *config.AlgorithmConfig, fields map[string]Value) {
		latencyAwareAlgorithmToFields(algo.LatencyAware, fields)
	},
	"multi_factor": func(algo *config.AlgorithmConfig, fields map[string]Value) {
		multiFactorAlgorithmToFields(algo.MultiFactor, fields)
	},
}

func (d *decompiler) algorithmToFields(algo *config.AlgorithmConfig) map[string]Value {
	fields := make(map[string]Value)
	if algo == nil {
		return fields
	}
	algorithmOnErrorToFields(algo, fields)
	if export, ok := algorithmFieldExporters[algo.Type]; ok {
		export(algo, fields)
	}
	return fields
}

func algorithmOnErrorToFields(algo *config.AlgorithmConfig, fields map[string]Value) {
	switch algo.Type {
	case "confidence", "ratings", "remom":
		return
	default:
		if algo.OnError != "" {
			fields["on_error"] = StringValue{V: algo.OnError}
		}
	}
}

func confidenceAlgorithmToFields(c *config.ConfidenceAlgorithmConfig, fields map[string]Value) {
	if c == nil {
		return
	}
	setStringValue(fields, "confidence_method", c.ConfidenceMethod)
	setFloatValue(fields, "threshold", c.Threshold)
	setStringValue(fields, "on_error", c.OnError)
	setStringValue(fields, "escalation_order", c.EscalationOrder)
	setFloatValue(fields, "cost_quality_tradeoff", c.CostQualityTradeoff)
	setStringValue(fields, "token_filter", c.TokenFilter)
	setStringValue(fields, "verifier_server_url", c.VerifierServerURL)
	setIntValue(fields, "verifier_timeout_seconds", c.VerifierTimeoutSeconds)
	if c.HybridWeights != nil {
		weights := map[string]Value{}
		setFloatValue(weights, "logprob_weight", c.HybridWeights.LogprobWeight)
		setFloatValue(weights, "margin_weight", c.HybridWeights.MarginWeight)
		fields["hybrid_weights"] = ObjectValue{Fields: weights}
	}
}

func ratingsAlgorithmToFields(r *config.RatingsAlgorithmConfig, fields map[string]Value) {
	if r == nil {
		return
	}
	setIntValue(fields, "max_concurrent", r.MaxConcurrent)
	setStringValue(fields, "on_error", r.OnError)
}

func remomAlgorithmToFields(r *config.ReMoMAlgorithmConfig, fields map[string]Value) {
	if r == nil {
		return
	}
	setIntArrayValue(fields, "breadth_schedule", r.BreadthSchedule)
	setStringValue(fields, "model_distribution", r.ModelDistribution)
	setFloatValue(fields, "temperature", r.Temperature)
	setBoolTrueValue(fields, "include_reasoning", r.IncludeReasoning)
	setStringValue(fields, "compaction_strategy", r.CompactionStrategy)
	setIntValue(fields, "compaction_tokens", r.CompactionTokens)
	setStringValue(fields, "synthesis_template", r.SynthesisTemplate)
	setIntValue(fields, "max_concurrent", r.MaxConcurrent)
	setStringValue(fields, "on_error", r.OnError)
	setIntValue(fields, "shuffle_seed", r.ShuffleSeed)
	setBoolTrueValue(fields, "include_intermediate_responses", r.IncludeIntermediateResponses)
	setIntValue(fields, "max_responses_per_round", r.MaxResponsesPerRound)
}

func eloAlgorithmToFields(e *config.EloSelectionConfig, fields map[string]Value) {
	if e == nil {
		return
	}
	setFloatValue(fields, "initial_rating", e.InitialRating)
	setFloatValue(fields, "k_factor", e.KFactor)
	setBoolTrueValue(fields, "category_weighted", e.CategoryWeighted)
	setFloatValue(fields, "decay_factor", e.DecayFactor)
	setIntValue(fields, "min_comparisons", e.MinComparisons)
	setFloatValue(fields, "cost_scaling_factor", e.CostScalingFactor)
	setStringValue(fields, "storage_path", e.StoragePath)
	setStringValue(fields, "auto_save_interval", e.AutoSaveInterval)
}

func routerDCAlgorithmToFields(r *config.RouterDCSelectionConfig, fields map[string]Value) {
	if r == nil {
		return
	}
	setFloatValue(fields, "temperature", r.Temperature)
	setIntValue(fields, "dimension_size", r.DimensionSize)
	setFloatValue(fields, "min_similarity", r.MinSimilarity)
	setBoolTrueValue(fields, "use_query_contrastive", r.UseQueryContrastive)
	setBoolTrueValue(fields, "use_model_contrastive", r.UseModelContrastive)
	setBoolTrueValue(fields, "require_descriptions", r.RequireDescriptions)
	setBoolTrueValue(fields, "use_capabilities", r.UseCapabilities)
}

func autoMixAlgorithmToFields(a *config.AutoMixSelectionConfig, fields map[string]Value) {
	if a == nil {
		return
	}
	setFloatValue(fields, "verification_threshold", a.VerificationThreshold)
	setIntValue(fields, "max_escalations", a.MaxEscalations)
	setBoolTrueValue(fields, "cost_aware_routing", a.CostAwareRouting)
	setFloatValue(fields, "cost_quality_tradeoff", a.CostQualityTradeoff)
	setFloatValue(fields, "discount_factor", a.DiscountFactor)
	setBoolTrueValue(fields, "use_logprob_verification", a.UseLogprobVerification)
}

func hybridAlgorithmToFields(h *config.HybridSelectionConfig, fields map[string]Value) {
	if h == nil {
		return
	}
	setFloatValue(fields, "elo_weight", h.EloWeight)
	setFloatValue(fields, "router_dc_weight", h.RouterDCWeight)
	setFloatValue(fields, "automix_weight", h.AutoMixWeight)
	setFloatValue(fields, "cost_weight", h.CostWeight)
	setFloatValue(fields, "quality_gap_threshold", h.QualityGapThreshold)
	setBoolTrueValue(fields, "normalize_scores", h.NormalizeScores)
}

func rlDrivenAlgorithmToFields(r *config.RLDrivenSelectionConfig, fields map[string]Value) {
	if r == nil {
		return
	}
	setFloatValue(fields, "exploration_rate", r.ExplorationRate)
	setFloatValue(fields, "exploration_decay", r.ExplorationDecay)
	setFloatValue(fields, "min_exploration", r.MinExploration)
	setBoolTrueValue(fields, "use_thompson_sampling", r.UseThompsonSampling)
	setBoolTrueValue(fields, "enable_personalization", r.EnablePersonalization)
	setFloatValue(fields, "personalization_blend", r.PersonalizationBlend)
	setFloatValue(fields, "session_context_weight", r.SessionContextWeight)
	setFloatValue(fields, "implicit_feedback_weight", r.ImplicitFeedbackWeight)
	setBoolTrueValue(fields, "cost_awareness", r.CostAwareness)
	setFloatValue(fields, "cost_weight", r.CostWeight)
	setStringValue(fields, "storage_path", r.StoragePath)
	setStringValue(fields, "auto_save_interval", r.AutoSaveInterval)
	setBoolTrueValue(fields, "use_router_r1_rewards", r.UseRouterR1Rewards)
	setFloatValue(fields, "cost_reward_alpha", r.CostRewardAlpha)
	setFloatValue(fields, "format_reward_penalty", r.FormatRewardPenalty)
	setBoolTrueValue(fields, "enable_llm_routing", r.EnableLLMRouting)
	setStringValue(fields, "router_r1_server_url", r.RouterR1ServerURL)
	setStringValue(fields, "llm_routing_fallback", r.LLMRoutingFallback)
	setBoolTrueValue(fields, "enable_multi_round_aggregation", r.EnableMultiRoundAggregation)
	setIntValue(fields, "max_aggregation_rounds", r.MaxAggregationRounds)
}

func gmtRouterAlgorithmToFields(g *config.GMTRouterSelectionConfig, fields map[string]Value) {
	if g == nil {
		return
	}
	setBoolTrueValue(fields, "enable_personalization", g.EnablePersonalization)
	setIntValue(fields, "history_sample_size", g.HistorySampleSize)
	setIntValue(fields, "embedding_dimension", g.EmbeddingDimension)
	setIntValue(fields, "num_gnn_layers", g.NumGNNLayers)
	setIntValue(fields, "attention_heads", g.AttentionHeads)
	setIntValue(fields, "min_interactions_for_personalization", g.MinInteractionsForPersonalization)
	setIntValue(fields, "max_interactions_per_user", g.MaxInteractionsPerUser)
	setStringArrayValue(fields, "feedback_types", g.FeedbackTypes)
	setStringValue(fields, "model_path", g.ModelPath)
	setStringValue(fields, "storage_path", g.StoragePath)
}

func latencyAwareAlgorithmToFields(l *config.LatencyAwareAlgorithmConfig, fields map[string]Value) {
	if l == nil {
		return
	}
	setIntValue(fields, "tpot_percentile", l.TPOTPercentile)
	setIntValue(fields, "ttft_percentile", l.TTFTPercentile)
	setStringValue(fields, "description", l.Description)
}

func multiFactorAlgorithmToFields(m *config.MultiFactorSelectionConfig, fields map[string]Value) {
	if m == nil {
		return
	}
	if m.Weights != nil {
		fields["weights"] = multiFactorWeightsValue(m.Weights)
	}
	if m.SLO != nil {
		fields["slo"] = multiFactorSLOValue(m.SLO)
	}
	setIntValue(fields, "latency_percentile", m.LatencyPercentile)
	setStringValue(fields, "on_no_candidates", m.OnNoCandidates)
}

func setStringValue(fields map[string]Value, key string, value string) {
	if value != "" {
		fields[key] = StringValue{V: value}
	}
}

func setIntValue(fields map[string]Value, key string, value int) {
	if value != 0 {
		fields[key] = IntValue{V: value}
	}
}

func setFloatValue(fields map[string]Value, key string, value float64) {
	if value != 0 {
		fields[key] = FloatValue{V: value}
	}
}

func setBoolTrueValue(fields map[string]Value, key string, value bool) {
	if value {
		fields[key] = BoolValue{V: true}
	}
}

func setStringArrayValue(fields map[string]Value, key string, values []string) {
	if len(values) > 0 {
		fields[key] = stringsToArray(values)
	}
}

func setIntArrayValue(fields map[string]Value, key string, values []int) {
	if len(values) == 0 {
		return
	}
	items := make([]Value, 0, len(values))
	for _, v := range values {
		items = append(items, IntValue{V: v})
	}
	fields[key] = ArrayValue{Items: items}
}

func multiFactorWeightsValue(weights *config.MultiFactorWeightsConfig) ObjectValue {
	fields := map[string]Value{}
	if weights == nil {
		return ObjectValue{Fields: fields}
	}
	setFloatValue(fields, "quality", weights.Quality)
	setFloatValue(fields, "latency", weights.Latency)
	setFloatValue(fields, "cost", weights.Cost)
	setFloatValue(fields, "load", weights.Load)
	return ObjectValue{Fields: fields}
}

func multiFactorSLOValue(slo *config.MultiFactorSLOConfig) ObjectValue {
	fields := map[string]Value{}
	if slo == nil {
		return ObjectValue{Fields: fields}
	}
	setFloatValue(fields, "max_tpot_ms", slo.MaxTPOTMs)
	setFloatValue(fields, "max_ttft_ms", slo.MaxTTFTMs)
	setFloatValue(fields, "max_cost_per_1m", slo.MaxCostPer1M)
	setIntValue(fields, "max_inflight", slo.MaxInflight)
	return ObjectValue{Fields: fields}
}
