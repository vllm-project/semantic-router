package dsl

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

type algorithmSubConfigCompiler func(*Compiler, *config.AlgorithmConfig, map[string]Value)

var algorithmSubConfigCompilers = map[string]algorithmSubConfigCompiler{
	"confidence": func(c *Compiler, algo *config.AlgorithmConfig, fields map[string]Value) {
		algo.Confidence = c.compileConfidenceAlgo(fields)
	},
	"ratings": func(c *Compiler, algo *config.AlgorithmConfig, fields map[string]Value) {
		algo.Ratings = c.compileRatingsAlgo(fields)
	},
	"remom": func(c *Compiler, algo *config.AlgorithmConfig, fields map[string]Value) {
		algo.ReMoM = c.compileReMoMAlgo(fields)
	},
	"router_dc": func(c *Compiler, algo *config.AlgorithmConfig, fields map[string]Value) {
		algo.RouterDC = c.compileRouterDCAlgo(fields)
	},
	"automix": func(c *Compiler, algo *config.AlgorithmConfig, fields map[string]Value) {
		algo.AutoMix = c.compileAutoMixAlgo(fields)
	},
	"hybrid": func(c *Compiler, algo *config.AlgorithmConfig, fields map[string]Value) {
		algo.Hybrid = c.compileHybridAlgo(fields)
	},
	"latency_aware": func(c *Compiler, algo *config.AlgorithmConfig, fields map[string]Value) {
		algo.LatencyAware = c.compileLatencyAwareAlgo(fields)
	},
	"multi_factor": func(c *Compiler, algo *config.AlgorithmConfig, fields map[string]Value) {
		algo.MultiFactor = c.compileMultiFactorAlgo(fields)
	},
	"static": func(*Compiler, *config.AlgorithmConfig, map[string]Value) {},
	"knn":    func(*Compiler, *config.AlgorithmConfig, map[string]Value) {},
	"kmeans": func(*Compiler, *config.AlgorithmConfig, map[string]Value) {},
	"mlp":    func(*Compiler, *config.AlgorithmConfig, map[string]Value) {},
	"svm":    func(*Compiler, *config.AlgorithmConfig, map[string]Value) {},
}

func (c *Compiler) compileAlgorithm(spec *AlgoSpec) *config.AlgorithmConfig {
	algo := &config.AlgorithmConfig{Type: spec.AlgoType}
	c.populateAlgorithmSubConfig(algo, spec)
	c.setAlgorithmTopLevelOnError(algo, spec.Fields)
	return algo
}

func (c *Compiler) populateAlgorithmSubConfig(algo *config.AlgorithmConfig, spec *AlgoSpec) {
	fn, ok := algorithmSubConfigCompilers[spec.AlgoType]
	if !ok {
		c.addError(spec.Pos, "unknown algorithm type %q", spec.AlgoType)
		return
	}
	fn(c, algo, spec.Fields)
}

func (c *Compiler) setAlgorithmTopLevelOnError(algo *config.AlgorithmConfig, fields map[string]Value) {
	switch algo.Type {
	case "confidence", "ratings", "remom":
		return
	default:
		if v, ok := getStringField(fields, "on_error"); ok {
			algo.OnError = v
		}
	}
}

func (c *Compiler) compileConfidenceAlgo(fields map[string]Value) *config.ConfidenceAlgorithmConfig {
	cfg := &config.ConfidenceAlgorithmConfig{}
	if v, ok := getStringField(fields, "confidence_method"); ok {
		cfg.ConfidenceMethod = v
	}
	if v, ok := getFloat64Field(fields, "threshold"); ok {
		cfg.Threshold = v
	}
	if v, ok := getStringField(fields, "on_error"); ok {
		cfg.OnError = v
	}
	if v, ok := getStringField(fields, "escalation_order"); ok {
		cfg.EscalationOrder = v
	}
	if v, ok := getFloat64Field(fields, "cost_quality_tradeoff"); ok {
		cfg.CostQualityTradeoff = v
	}
	if v, ok := getStringField(fields, "token_filter"); ok {
		cfg.TokenFilter = v
	}
	if v, ok := getStringField(fields, "verifier_server_url"); ok {
		cfg.VerifierServerURL = v
	}
	if v, ok := getIntField(fields, "verifier_timeout_seconds"); ok {
		cfg.VerifierTimeoutSeconds = v
	}
	cfg.HybridWeights = parseHybridWeights(fields)
	return cfg
}

func parseHybridWeights(fields map[string]Value) *config.HybridWeightsConfig {
	obj, ok := fields["hybrid_weights"].(ObjectValue)
	if !ok {
		return nil
	}
	hw := &config.HybridWeightsConfig{}
	if v, ok := getFloat64Field(obj.Fields, "logprob_weight"); ok {
		hw.LogprobWeight = v
	}
	if v, ok := getFloat64Field(obj.Fields, "margin_weight"); ok {
		hw.MarginWeight = v
	}
	return hw
}

func (c *Compiler) compileRatingsAlgo(fields map[string]Value) *config.RatingsAlgorithmConfig {
	cfg := &config.RatingsAlgorithmConfig{}
	if v, ok := getIntField(fields, "max_concurrent"); ok {
		cfg.MaxConcurrent = v
	}
	if v, ok := getStringField(fields, "on_error"); ok {
		cfg.OnError = v
	}
	return cfg
}

func (c *Compiler) compileReMoMAlgo(fields map[string]Value) *config.ReMoMAlgorithmConfig {
	cfg := &config.ReMoMAlgorithmConfig{}
	fillReMoMPlanningFields(cfg, fields)
	fillReMoMRuntimeFields(cfg, fields)
	fillReMoMResponseFields(cfg, fields)
	return cfg
}

func fillReMoMPlanningFields(cfg *config.ReMoMAlgorithmConfig, fields map[string]Value) {
	if v, ok := getIntArrayField(fields, "breadth_schedule"); ok {
		cfg.BreadthSchedule = v
	}
	if v, ok := getStringField(fields, "model_distribution"); ok {
		cfg.ModelDistribution = v
	}
	if v, ok := getFloat64Field(fields, "temperature"); ok {
		cfg.Temperature = v
	}
}

func fillReMoMRuntimeFields(cfg *config.ReMoMAlgorithmConfig, fields map[string]Value) {
	if v, ok := getBoolField(fields, "include_reasoning"); ok {
		cfg.IncludeReasoning = v
	}
	if v, ok := getStringField(fields, "compaction_strategy"); ok {
		cfg.CompactionStrategy = v
	}
	if v, ok := getIntField(fields, "compaction_tokens"); ok {
		cfg.CompactionTokens = v
	}
	if v, ok := getStringField(fields, "synthesis_template"); ok {
		cfg.SynthesisTemplate = v
	}
	if v, ok := getIntField(fields, "max_concurrent"); ok {
		cfg.MaxConcurrent = v
	}
	if v, ok := getStringField(fields, "on_error"); ok {
		cfg.OnError = v
	}
}

func fillReMoMResponseFields(cfg *config.ReMoMAlgorithmConfig, fields map[string]Value) {
	if v, ok := getBoolField(fields, "include_intermediate_responses"); ok {
		cfg.IncludeIntermediateResponses = v
	}
	if v, ok := getIntField(fields, "shuffle_seed"); ok {
		cfg.ShuffleSeed = v
	}
	if v, ok := getIntField(fields, "max_responses_per_round"); ok {
		cfg.MaxResponsesPerRound = v
	}
}

func (c *Compiler) compileRouterDCAlgo(fields map[string]Value) *config.RouterDCSelectionConfig {
	cfg := &config.RouterDCSelectionConfig{}
	if v, ok := getFloat64Field(fields, "temperature"); ok {
		cfg.Temperature = v
	}
	if v, ok := getIntField(fields, "dimension_size"); ok {
		cfg.DimensionSize = v
	}
	if v, ok := getFloat64Field(fields, "min_similarity"); ok {
		cfg.MinSimilarity = v
	}
	if v, ok := getBoolField(fields, "use_query_contrastive"); ok {
		cfg.UseQueryContrastive = v
	}
	if v, ok := getBoolField(fields, "use_model_contrastive"); ok {
		cfg.UseModelContrastive = v
	}
	if v, ok := getBoolField(fields, "require_descriptions"); ok {
		cfg.RequireDescriptions = v
	}
	if v, ok := getBoolField(fields, "use_capabilities"); ok {
		cfg.UseCapabilities = v
	}
	return cfg
}

func (c *Compiler) compileAutoMixAlgo(fields map[string]Value) *config.AutoMixSelectionConfig {
	cfg := &config.AutoMixSelectionConfig{}
	if v, ok := getFloat64Field(fields, "verification_threshold"); ok {
		cfg.VerificationThreshold = v
	}
	if v, ok := getIntField(fields, "max_escalations"); ok {
		cfg.MaxEscalations = v
	}
	if v, ok := getBoolField(fields, "cost_aware_routing"); ok {
		cfg.CostAwareRouting = v
	}
	if v, ok := getFloat64Field(fields, "cost_quality_tradeoff"); ok {
		cfg.CostQualityTradeoff = v
	}
	if v, ok := getFloat64Field(fields, "discount_factor"); ok {
		cfg.DiscountFactor = v
	}
	if v, ok := getBoolField(fields, "use_logprob_verification"); ok {
		cfg.UseLogprobVerification = v
	}
	return cfg
}

func (c *Compiler) compileHybridAlgo(fields map[string]Value) *config.HybridSelectionConfig {
	cfg := &config.HybridSelectionConfig{}
	if v, ok := getFloat64Field(fields, "experience_weight"); ok {
		cfg.ExperienceWeight = v
	}
	if v, ok := getFloat64Field(fields, "router_dc_weight"); ok {
		cfg.RouterDCWeight = v
	}
	if v, ok := getFloat64Field(fields, "automix_weight"); ok {
		cfg.AutoMixWeight = v
	}
	if v, ok := getFloat64Field(fields, "cost_weight"); ok {
		cfg.CostWeight = v
	}
	if v, ok := getFloat64Field(fields, "quality_gap_threshold"); ok {
		cfg.QualityGapThreshold = v
	}
	if v, ok := getBoolField(fields, "normalize_scores"); ok {
		cfg.NormalizeScores = v
	}
	return cfg
}

func (c *Compiler) compileLatencyAwareAlgo(fields map[string]Value) *config.LatencyAwareAlgorithmConfig {
	cfg := &config.LatencyAwareAlgorithmConfig{}
	if v, ok := getIntField(fields, "tpot_percentile"); ok {
		cfg.TPOTPercentile = v
	}
	if v, ok := getIntField(fields, "ttft_percentile"); ok {
		cfg.TTFTPercentile = v
	}
	if v, ok := getStringField(fields, "description"); ok {
		cfg.Description = v
	}
	return cfg
}

func (c *Compiler) compileMultiFactorAlgo(fields map[string]Value) *config.MultiFactorSelectionConfig {
	cfg := &config.MultiFactorSelectionConfig{}
	cfg.Weights = parseMultiFactorWeights(fields)
	cfg.SLO = parseMultiFactorSLO(fields)
	if v, ok := getIntField(fields, "latency_percentile"); ok {
		cfg.LatencyPercentile = v
	}
	if v, ok := getStringField(fields, "on_no_candidates"); ok {
		cfg.OnNoCandidates = v
	}
	return cfg
}

func parseMultiFactorWeights(fields map[string]Value) *config.MultiFactorWeightsConfig {
	weights, ok := fields["weights"].(ObjectValue)
	if !ok {
		return nil
	}
	cfg := &config.MultiFactorWeightsConfig{}
	if v, ok := getFloat64Field(weights.Fields, "quality"); ok {
		cfg.Quality = v
	}
	if v, ok := getFloat64Field(weights.Fields, "latency"); ok {
		cfg.Latency = v
	}
	if v, ok := getFloat64Field(weights.Fields, "cost"); ok {
		cfg.Cost = v
	}
	if v, ok := getFloat64Field(weights.Fields, "load"); ok {
		cfg.Load = v
	}
	return cfg
}

func parseMultiFactorSLO(fields map[string]Value) *config.MultiFactorSLOConfig {
	slo, ok := fields["slo"].(ObjectValue)
	if !ok {
		return nil
	}
	cfg := &config.MultiFactorSLOConfig{}
	if v, ok := getFloat64Field(slo.Fields, "max_tpot_ms"); ok {
		cfg.MaxTPOTMs = v
	}
	if v, ok := getFloat64Field(slo.Fields, "max_ttft_ms"); ok {
		cfg.MaxTTFTMs = v
	}
	if v, ok := getFloat64Field(slo.Fields, "max_cost_per_1m"); ok {
		cfg.MaxCostPer1M = v
	}
	if v, ok := getIntField(slo.Fields, "max_inflight"); ok {
		cfg.MaxInflight = v
	}
	return cfg
}
