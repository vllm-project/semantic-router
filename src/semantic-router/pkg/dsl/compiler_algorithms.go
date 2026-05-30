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
	"elo": func(c *Compiler, algo *config.AlgorithmConfig, fields map[string]Value) {
		algo.Elo = c.compileEloAlgo(fields)
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
	"rl_driven": func(c *Compiler, algo *config.AlgorithmConfig, fields map[string]Value) {
		algo.RLDriven = c.compileRLDrivenAlgo(fields)
	},
	"gmtrouter": func(c *Compiler, algo *config.AlgorithmConfig, fields map[string]Value) {
		algo.GMTRouter = c.compileGMTRouterAlgo(fields)
	},
	"latency_aware": func(c *Compiler, algo *config.AlgorithmConfig, fields map[string]Value) {
		algo.LatencyAware = c.compileLatencyAwareAlgo(fields)
	},
	"static": func(*Compiler, *config.AlgorithmConfig, map[string]Value) {},
	"knn":    func(*Compiler, *config.AlgorithmConfig, map[string]Value) {},
	"kmeans": func(*Compiler, *config.AlgorithmConfig, map[string]Value) {},
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
	if v, ok := getIntArrayField(fields, "breadth_schedule"); ok {
		cfg.BreadthSchedule = v
	}
	if v, ok := getStringField(fields, "model_distribution"); ok {
		cfg.ModelDistribution = v
	}
	if v, ok := getFloat64Field(fields, "temperature"); ok {
		cfg.Temperature = v
	}
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
	if v, ok := getBoolField(fields, "include_intermediate_responses"); ok {
		cfg.IncludeIntermediateResponses = v
	}
	return cfg
}

func (c *Compiler) compileEloAlgo(fields map[string]Value) *config.EloSelectionConfig {
	cfg := &config.EloSelectionConfig{}
	if v, ok := getFloat64Field(fields, "initial_rating"); ok {
		cfg.InitialRating = v
	}
	if v, ok := getFloat64Field(fields, "k_factor"); ok {
		cfg.KFactor = v
	}
	if v, ok := getBoolField(fields, "category_weighted"); ok {
		cfg.CategoryWeighted = v
	}
	if v, ok := getFloat64Field(fields, "decay_factor"); ok {
		cfg.DecayFactor = v
	}
	if v, ok := getIntField(fields, "min_comparisons"); ok {
		cfg.MinComparisons = v
	}
	if v, ok := getFloat64Field(fields, "cost_scaling_factor"); ok {
		cfg.CostScalingFactor = v
	}
	if v, ok := getStringField(fields, "storage_path"); ok {
		cfg.StoragePath = v
	}
	return cfg
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
	return cfg
}

func (c *Compiler) compileHybridAlgo(fields map[string]Value) *config.HybridSelectionConfig {
	cfg := &config.HybridSelectionConfig{}
	if v, ok := getFloat64Field(fields, "elo_weight"); ok {
		cfg.EloWeight = v
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
	return cfg
}

func (c *Compiler) compileRLDrivenAlgo(fields map[string]Value) *config.RLDrivenSelectionConfig {
	cfg := &config.RLDrivenSelectionConfig{}
	if v, ok := getFloat64Field(fields, "exploration_rate"); ok {
		cfg.ExplorationRate = v
	}
	if v, ok := getBoolField(fields, "use_thompson_sampling"); ok {
		cfg.UseThompsonSampling = v
	}
	if v, ok := getBoolField(fields, "enable_personalization"); ok {
		cfg.EnablePersonalization = v
	}
	return cfg
}

func (c *Compiler) compileGMTRouterAlgo(fields map[string]Value) *config.GMTRouterSelectionConfig {
	cfg := &config.GMTRouterSelectionConfig{}
	if v, ok := getBoolField(fields, "enable_personalization"); ok {
		cfg.EnablePersonalization = v
	}
	if v, ok := getIntField(fields, "history_sample_size"); ok {
		cfg.HistorySampleSize = v
	}
	if v, ok := getStringField(fields, "model_path"); ok {
		cfg.ModelPath = v
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
	return cfg
}
