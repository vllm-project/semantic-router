package dsl

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

type algorithmFieldExporter func(*config.AlgorithmConfig, map[string]Value)

var algorithmFieldExporters = map[string]algorithmFieldExporter{
	config.DecisionAlgorithmEscalate: escalateAlgorithmToFields,
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
	"multi_factor": func(algo *config.AlgorithmConfig, fields map[string]Value) {
		multiFactorAlgorithmToFields(algo.MultiFactor, fields)
	},
	"systemone": func(algo *config.AlgorithmConfig, fields map[string]Value) {
		systemOneAlgorithmToFields(algo.SystemOne, fields)
	},
}

func systemOneAlgorithmToFields(
	selector *config.SystemOneSelectionConfig,
	fields map[string]Value,
) {
	if selector == nil {
		return
	}
	selectorFields := map[string]Value{}
	setStringValue(selectorFields, "deployment", selector.Deployment)
	setStringValue(selectorFields, "instructions", selector.Instructions)
	setIntValue(selectorFields, "timeout_ms", selector.TimeoutMs)
	if len(selector.Candidates) > 0 {
		candidates := map[string]Value{}
		for name, description := range selector.Candidates {
			candidates[name] = StringValue{V: description}
		}
		selectorFields["candidates"] = ObjectValue{Fields: candidates}
	}
	fields["systemone"] = ObjectValue{Fields: selectorFields}
}

func (d *decompiler) algorithmToFields(algo *config.AlgorithmConfig) map[string]Value {
	fields := make(map[string]Value)
	if algo == nil {
		return fields
	}
	setIntValue(fields, "minimum_candidates", algo.MinimumCandidates)
	setStringValue(fields, "on_error", algo.OnError)
	algorithmBudgetToFields(algo.Budget, fields)
	if export, ok := algorithmFieldExporters[algo.Type]; ok {
		export(algo, fields)
	}
	return fields
}

func ratingsAlgorithmToFields(r *config.RatingsAlgorithmConfig, fields map[string]Value) {
	if r == nil {
		return
	}
	setIntValue(fields, "max_concurrent", r.MaxConcurrent)
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
	if r.MaxCompletionTokens != nil {
		setIntValue(fields, "max_completion_tokens", *r.MaxCompletionTokens)
	}
	setIntValue(fields, "round_timeout_seconds", r.RoundTimeoutSeconds)
	setIntValue(fields, "min_successful_responses", r.MinSuccessfulResponses)
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
	setStringValue(fields, "analysis_mode", f.AnalysisMode)
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
	setStringValue(fields, "quorum_failure_policy", string(f.QuorumFailurePolicy))
	setStringValue(fields, "quorum_fallback_target", f.QuorumFallbackTarget)
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
	plannerFields := make(map[string]Value)
	setStringValue(plannerFields, "model", w.Planner.Model)
	setIntValue(plannerFields, "max_completion_tokens", w.Planner.MaxCompletionTokens)
	if len(plannerFields) > 0 {
		fields["planner"] = ObjectValue{Fields: plannerFields}
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
	if m.Quality != nil {
		fields["quality"] = qualityEvidenceValue(m.Quality)
	}
	if m.Objective != nil {
		fields["objective"] = multiFactorObjectiveValue(m.Objective)
	}
	if m.ExpectedOutputTokens != nil {
		fields["expected_output_tokens"] = IntValue{V: *m.ExpectedOutputTokens}
	}
	setStringValue(fields, "latency_metric", m.LatencyMetric)
	setIntValue(fields, "latency_percentile", m.LatencyPercentile)
	setStringValue(fields, "on_no_candidates", m.OnNoCandidates)
}

func qualityEvidenceValue(quality *config.QualityEvidenceConfig) ObjectValue {
	fields := map[string]Value{}
	if quality == nil {
		return ObjectValue{Fields: fields}
	}
	setStringValue(fields, "index", quality.Index)
	setStringValue(fields, "on_missing", quality.OnMissing)
	setFloatValue(fields, "min_coverage", quality.MinCoverage)
	if quality.MinScore != nil {
		fields["min_score"] = FloatValue{V: *quality.MinScore}
	}
	return ObjectValue{Fields: fields}
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
