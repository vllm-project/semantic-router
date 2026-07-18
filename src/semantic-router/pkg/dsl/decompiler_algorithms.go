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
	"fusion": func(algo *config.AlgorithmConfig, fields map[string]Value) {
		fusionAlgorithmToFields(algo.Fusion, fields)
	},
	"workflows": func(algo *config.AlgorithmConfig, fields map[string]Value) {
		workflowsAlgorithmToFields(algo.Workflows, fields)
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
	setStringValue(fields, "synthesis_model", r.SynthesisModel)
	setIntValue(fields, "max_concurrent", r.MaxConcurrent)
	setIntValue(fields, "round_timeout_seconds", r.RoundTimeoutSeconds)
	setIntValue(fields, "min_successful_responses", r.MinSuccessfulResponses)
	setStringValue(fields, "on_error", r.OnError)
	setIntValue(fields, "shuffle_seed", r.ShuffleSeed)
	setBoolTrueValue(fields, "include_intermediate_responses", r.IncludeIntermediateResponses)
	setIntValue(fields, "max_responses_per_round", r.MaxResponsesPerRound)
}

func fusionAlgorithmToFields(f *config.FusionAlgorithmConfig, fields map[string]Value) {
	if f == nil {
		return
	}
	setStringValue(fields, "model", f.Model)
	if len(f.AnalysisModels) > 0 {
		fields["analysis_models"] = stringsToArray(f.AnalysisModels)
	}
	setIntValue(fields, "max_concurrent", f.MaxConcurrent)
	setIntValue(fields, "max_completion_tokens", f.MaxCompletionTokens)
	setIntValue(fields, "round_timeout_seconds", f.RoundTimeoutSeconds)
	setIntValue(fields, "min_successful_responses", f.MinSuccessfulResponses)
	if f.Temperature != nil {
		fields["temperature"] = FloatValue{V: *f.Temperature}
	}
	if f.IncludeAnalysis != nil {
		fields["include_analysis"] = BoolValue{V: *f.IncludeAnalysis}
	}
	setStringValue(fields, "on_error", f.OnError)
	setStringValue(fields, "analysis_template", f.AnalysisTemplate)
	setStringValue(fields, "synthesis_template", f.SynthesisTemplate)
	setStringValue(fields, "judge_prompt_version", f.JudgePromptVersion)
	if f.IncludeIntermediateResponses != nil {
		fields["include_intermediate_responses"] = BoolValue{V: *f.IncludeIntermediateResponses}
	}
}

func workflowsAlgorithmToFields(w *config.WorkflowsAlgorithmConfig, fields map[string]Value) {
	if w == nil {
		return
	}
	setStringValue(fields, "mode", w.Mode)
	setStringValue(fields, "template", w.Template)
	if len(w.Roles) > 0 {
		fields["roles"] = workflowRolesValue(w.Roles)
	}
	if !w.Final.IsZero() {
		fields["final"] = workflowFinalValue(w.Final)
	}
	if w.Planner.Model != "" {
		fields["planner"] = ObjectValue{Fields: map[string]Value{
			"model": StringValue{V: w.Planner.Model},
		}}
	}
	setIntValue(fields, "max_steps", w.MaxSteps)
	setIntValue(fields, "max_parallel", w.MaxParallel)
	setIntValue(fields, "max_completion_tokens", w.MaxCompletionTokens)
	setIntValue(fields, "round_timeout_seconds", w.RoundTimeoutSeconds)
	setIntValue(fields, "min_successful_responses", w.MinSuccessfulResponses)
	if w.Temperature != nil {
		fields["temperature"] = FloatValue{V: *w.Temperature}
	}
	if w.IncludeIntermediateResponses != nil {
		fields["include_intermediate_responses"] = BoolValue{V: *w.IncludeIntermediateResponses}
	}
	setStringValue(fields, "on_error", w.OnError)
}

func workflowRolesValue(roles []config.WorkflowRoleConfig) ArrayValue {
	items := make([]Value, 0, len(roles))
	for _, role := range roles {
		fields := map[string]Value{}
		setStringValue(fields, "name", role.Name)
		if len(role.Models) > 0 {
			fields["models"] = stringsToArray(role.Models)
		}
		setStringValue(fields, "prompt", role.Prompt)
		items = append(items, ObjectValue{Fields: fields})
	}
	return ArrayValue{Items: items}
}

func workflowFinalValue(final config.WorkflowFinalConfig) ObjectValue {
	fields := map[string]Value{}
	setStringValue(fields, "model", final.Model)
	setStringValue(fields, "prompt", final.Prompt)
	return ObjectValue{Fields: fields}
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
	setFloatValue(fields, "experience_weight", h.ExperienceWeight)
	setFloatValue(fields, "router_dc_weight", h.RouterDCWeight)
	setFloatValue(fields, "automix_weight", h.AutoMixWeight)
	setFloatValue(fields, "cost_weight", h.CostWeight)
	setFloatValue(fields, "quality_gap_threshold", h.QualityGapThreshold)
	setBoolTrueValue(fields, "normalize_scores", h.NormalizeScores)
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
