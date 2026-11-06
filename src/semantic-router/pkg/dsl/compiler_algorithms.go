package dsl

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

type algorithmSubConfigCompiler func(*Compiler, *config.AlgorithmConfig, map[string]Value)

var algorithmSubConfigCompilers = map[string]algorithmSubConfigCompiler{
	config.DecisionAlgorithmEscalate: compileEscalateAlgorithm,
	"ratings": func(c *Compiler, algo *config.AlgorithmConfig, fields map[string]Value) {
		algo.Ratings = c.compileRatingsAlgo(fields)
	},
	"remom": func(c *Compiler, algo *config.AlgorithmConfig, fields map[string]Value) {
		algo.ReMoM = c.compileReMoMAlgo(fields)
	},
	"fusion": func(c *Compiler, algo *config.AlgorithmConfig, fields map[string]Value) {
		algo.Fusion = c.compileFusionAlgo(fields)
	},
	"workflows": func(c *Compiler, algo *config.AlgorithmConfig, fields map[string]Value) {
		algo.Workflows = c.compileWorkflowsAlgo(fields)
	},
	"multi_factor": func(c *Compiler, algo *config.AlgorithmConfig, fields map[string]Value) {
		algo.MultiFactor = c.compileMultiFactorAlgo(fields)
	},
	"systemone": func(c *Compiler, algo *config.AlgorithmConfig, fields map[string]Value) {
		algo.SystemOne = c.compileSystemOneAlgo(fields)
	},
	"static": func(*Compiler, *config.AlgorithmConfig, map[string]Value) {},
	"random": func(*Compiler, *config.AlgorithmConfig, map[string]Value) {},
}

func (c *Compiler) compileSystemOneAlgo(
	fields map[string]Value,
) *config.SystemOneSelectionConfig {
	cfg := &config.SystemOneSelectionConfig{}
	selector, ok := fields["systemone"].(ObjectValue)
	if !ok {
		return cfg
	}
	if value, ok := getStringField(selector.Fields, "deployment"); ok {
		cfg.Deployment = value
	}
	if value, ok := getStringField(selector.Fields, "instructions"); ok {
		cfg.Instructions = value
	}
	if value, ok := getIntField(selector.Fields, "timeout_ms"); ok {
		cfg.TimeoutMs = value
	}
	if candidates, ok := selector.Fields["candidates"].(ObjectValue); ok {
		cfg.Candidates = make(map[string]string, len(candidates.Fields))
		for name := range candidates.Fields {
			if value, ok := getStringField(candidates.Fields, name); ok {
				cfg.Candidates[name] = value
			}
		}
	}
	return cfg
}

func (c *Compiler) compileAlgorithm(spec *AlgoSpec) *config.AlgorithmConfig {
	algo := &config.AlgorithmConfig{Type: spec.AlgoType}
	if minimum, ok := getIntField(spec.Fields, "minimum_candidates"); ok {
		algo.MinimumCandidates = minimum
	}
	c.populateAlgorithmSubConfig(algo, spec)
	c.compileAlgorithmBudget(spec, algo)
	if v, ok := getStringField(spec.Fields, "on_error"); ok {
		algo.OnError = v
	}
	if (algo.Budget != nil || algo.OnError != "") && !config.IsExecutorAlgorithmType(spec.AlgoType) {
		c.addError(spec.Pos, "budget and on_error are executor settings; %s is a selector", spec.AlgoType)
	}
	if algo.ReMoM != nil {
		if err := config.ValidateReMoMAlgorithmConfig(algo.ReMoM); err != nil {
			c.addError(spec.Pos, "invalid remom algorithm: %v", err)
		}
	}
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

func (c *Compiler) compileRatingsAlgo(fields map[string]Value) *config.RatingsAlgorithmConfig {
	cfg := &config.RatingsAlgorithmConfig{}
	if v, ok := getIntField(fields, "max_concurrent"); ok {
		cfg.MaxConcurrent = v
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
	if v, ok := getStringField(fields, "synthesis_model"); ok {
		cfg.SynthesisModel = v
	}
	if v, ok := getIntField(fields, "max_concurrent"); ok {
		cfg.MaxConcurrent = v
	}
	if v, ok := getIntField(fields, "max_completion_tokens"); ok {
		cfg.MaxCompletionTokens = &v
	}
	if v, ok := getIntField(fields, "round_timeout_seconds"); ok {
		cfg.RoundTimeoutSeconds = v
	}
	if v, ok := getIntField(fields, "min_successful_responses"); ok {
		cfg.MinSuccessfulResponses = v
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

func (c *Compiler) compileFusionAlgo(fields map[string]Value) *config.FusionAlgorithmConfig {
	cfg := &config.FusionAlgorithmConfig{}
	fillFusionModelFields(cfg, fields)
	fillFusionRuntimeFields(cfg, fields)
	fillFusionResponseFields(cfg, fields)
	fillFusionPromptFields(cfg, fields)
	return cfg
}

func fillFusionModelFields(cfg *config.FusionAlgorithmConfig, fields map[string]Value) {
	if v, ok := getStringField(fields, "model"); ok {
		cfg.Model = v
	}
	if v, ok := getStringArrayField(fields, "analysis_models"); ok {
		cfg.AnalysisModels = v
	}
}

func fillFusionRuntimeFields(cfg *config.FusionAlgorithmConfig, fields map[string]Value) {
	if v, ok := getStringField(fields, "analysis_mode"); ok {
		cfg.AnalysisMode = v
	}
	if v, ok := getIntField(fields, "max_concurrent"); ok {
		cfg.MaxConcurrent = v
	}
	if v, ok := getIntField(fields, "max_completion_tokens"); ok {
		cfg.MaxCompletionTokens = v
	}
	if v, ok := getIntField(fields, "round_timeout_seconds"); ok {
		cfg.RoundTimeoutSeconds = v
	}
	if v, ok := getIntField(fields, "min_successful_responses"); ok {
		cfg.MinSuccessfulResponses = v
	}
	if v, ok := getFloat64Field(fields, "temperature"); ok {
		cfg.Temperature = &v
	}
	if v, ok := getStringField(fields, "quorum_failure_policy"); ok {
		cfg.QuorumFailurePolicy = config.FusionQuorumFailurePolicy(v)
	}
	if v, ok := getStringField(fields, "quorum_fallback_target"); ok {
		cfg.QuorumFallbackTarget = v
	}
}

func fillFusionResponseFields(cfg *config.FusionAlgorithmConfig, fields map[string]Value) {
	if v, ok := getBoolField(fields, "include_analysis"); ok {
		cfg.IncludeAnalysis = &v
	}
	if v, ok := getBoolField(fields, "include_intermediate_responses"); ok {
		cfg.IncludeIntermediateResponses = &v
	}
}

func fillFusionPromptFields(cfg *config.FusionAlgorithmConfig, fields map[string]Value) {
	if v, ok := getStringField(fields, "analysis_template"); ok {
		cfg.AnalysisTemplate = v
	}
	if v, ok := getStringField(fields, "synthesis_template"); ok {
		cfg.SynthesisTemplate = v
	}
	if v, ok := getStringField(fields, "judge_prompt_version"); ok {
		cfg.JudgePromptVersion = v
	}
}

func (c *Compiler) compileWorkflowsAlgo(fields map[string]Value) *config.WorkflowsAlgorithmConfig {
	cfg := &config.WorkflowsAlgorithmConfig{}
	compileWorkflowIdentity(fields, cfg)
	cfg.Roles = compileWorkflowRoles(fields)
	cfg.Final = compileWorkflowFinalField(fields)
	compileWorkflowPlanner(fields, cfg)
	compileWorkflowLimits(fields, cfg)
	return cfg
}

func compileWorkflowIdentity(fields map[string]Value, cfg *config.WorkflowsAlgorithmConfig) {
	if v, ok := getStringField(fields, "mode"); ok {
		cfg.Mode = v
	}
	if v, ok := getStringField(fields, "template"); ok {
		cfg.Template = v
	}
}

func compileWorkflowFinalField(fields map[string]Value) config.WorkflowFinalConfig {
	if final, ok := fields["final"].(ObjectValue); ok {
		return compileWorkflowFinal(final)
	}
	return config.WorkflowFinalConfig{}
}

func compileWorkflowPlanner(fields map[string]Value, cfg *config.WorkflowsAlgorithmConfig) {
	if planner, ok := fields["planner"].(ObjectValue); ok {
		if v, ok := getStringField(planner.Fields, "model"); ok {
			cfg.Planner.Model = v
		}
		if v, ok := getIntField(planner.Fields, "max_completion_tokens"); ok {
			cfg.Planner.MaxCompletionTokens = v
		}
	}
	if v, ok := getStringField(fields, "planner.model"); ok {
		cfg.Planner.Model = v
	}
}

func compileWorkflowLimits(fields map[string]Value, cfg *config.WorkflowsAlgorithmConfig) {
	if v, ok := getIntField(fields, "max_steps"); ok {
		cfg.MaxSteps = v
	}
	if v, ok := getIntField(fields, "max_parallel"); ok {
		cfg.MaxParallel = v
	}
	if v, ok := getIntField(fields, "max_completion_tokens"); ok {
		cfg.MaxCompletionTokens = v
	}
	if v, ok := getIntField(fields, "round_timeout_seconds"); ok {
		cfg.RoundTimeoutSeconds = v
	}
	if v, ok := getIntField(fields, "min_successful_responses"); ok {
		cfg.MinSuccessfulResponses = v
	}
	if v, ok := getFloat64Field(fields, "temperature"); ok {
		cfg.Temperature = &v
	}
	if v, ok := getBoolField(fields, "include_intermediate_responses"); ok {
		cfg.IncludeIntermediateResponses = &v
	}
}

func compileWorkflowRoles(fields map[string]Value) []config.WorkflowRoleConfig {
	raw, ok := fields["roles"].(ArrayValue)
	if !ok {
		return nil
	}
	roles := make([]config.WorkflowRoleConfig, 0, len(raw.Items))
	for _, item := range raw.Items {
		roleObj, ok := item.(ObjectValue)
		if !ok {
			continue
		}
		role := config.WorkflowRoleConfig{}
		if v, ok := getStringField(roleObj.Fields, "name"); ok {
			role.Name = v
		}
		if models, ok := getStringArrayField(roleObj.Fields, "models"); ok {
			role.Models = models
		}
		if v, ok := getStringField(roleObj.Fields, "prompt"); ok {
			role.Prompt = v
		}
		roles = append(roles, role)
	}
	return roles
}

func compileWorkflowFinal(final ObjectValue) config.WorkflowFinalConfig {
	cfg := config.WorkflowFinalConfig{}
	if v, ok := getStringField(final.Fields, "model"); ok {
		cfg.Model = v
	}
	if v, ok := getStringField(final.Fields, "prompt"); ok {
		cfg.Prompt = v
	}
	return cfg
}

func (c *Compiler) compileMultiFactorAlgo(fields map[string]Value) *config.MultiFactorSelectionConfig {
	cfg, err := decodeMultiFactorFields(fields)
	if err != nil {
		c.errors = append(c.errors, err)
	}
	return cfg
}
