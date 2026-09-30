package looper

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (l *FusionLooper) resolveFusionExecutionConfig(req *Request) fusionExecutionConfig {
	cfg := fusionExecutionConfig{
		AnalysisMode:                 config.FusionAnalysisModeSeparate,
		IncludeAnalysis:              true,
		IncludeIntermediateResponses: true,
	}
	recipeOwnsExecution := req.Algorithm != nil && req.Algorithm.Type == config.DecisionAlgorithmFusion

	algorithmHasAnalysisModels := req.Algorithm != nil &&
		req.Algorithm.Fusion != nil &&
		len(req.Algorithm.Fusion.AnalysisModels) > 0
	if algorithmHasAnalysisModels || req.Algorithm != nil && req.Algorithm.Fusion != nil {
		mergeFusionAlgorithmConfig(&cfg, req.Algorithm.Fusion)
	}
	if len(req.ModelRefs) > 0 && !algorithmHasAnalysisModels {
		cfg.AnalysisModels = modelRefsToNames(req.ModelRefs)
	}
	if len(cfg.AnalysisModels) == 0 {
		cfg.AnalysisModels = modelRefsToNames(req.ModelRefs)
	}
	if req.Fusion != nil && fusionRequestEnabled(req.Fusion) {
		if recipeOwnsExecution {
			// The selected decision owns every execution control. Requests may
			// carry only call-level choices for exposing existing trace data.
			mergeFusionTraceVisibility(
				&cfg,
				req.Fusion.IncludeAnalysis,
				req.Fusion.IncludeIntermediateResponses,
			)
		} else {
			mergeFusionRequestConfig(&cfg, req.Fusion)
		}
	}
	return normalizeFusionExecutionConfig(cfg)
}

func normalizeFusionExecutionConfig(cfg fusionExecutionConfig) fusionExecutionConfig {
	cfg.AnalysisMode = config.EffectiveFusionAnalysisMode(cfg.AnalysisMode)
	cfg.OnError = strings.TrimSpace(cfg.OnError)
	if cfg.OnError == "" {
		cfg.OnError = config.FusionOnErrorSkip
	}
	if cfg.QuorumFailurePolicy == "" {
		cfg.QuorumFailurePolicy = config.FusionQuorumFailurePolicyFail
	}
	if cfg.JudgePromptVersion == "" {
		cfg.JudgePromptVersion = config.DefaultFusionJudgePromptVersion
	}
	cfg.AnalysisModels = normalizeModelNames(cfg.AnalysisModels)
	cfg.AnalysisOverrides = normalizeFusionAnalysisOverrides(cfg.AnalysisModels, cfg.AnalysisOverrides)
	if cfg.MaxConcurrent <= 0 || cfg.MaxConcurrent > len(cfg.AnalysisModels) {
		cfg.MaxConcurrent = len(cfg.AnalysisModels)
	}
	if cfg.MinSuccessfulResponses <= 0 {
		cfg.MinSuccessfulResponses = len(cfg.AnalysisModels)
	}
	applyGroundingDefaults(&cfg)
	return cfg
}

// applyGroundingDefaults fills in grounding defaults when grounding is enabled.
func applyGroundingDefaults(cfg *fusionExecutionConfig) {
	if !cfg.GroundingEnabled {
		return
	}
	if cfg.GroundingReference == "" {
		cfg.GroundingReference = config.FusionGroundingReferenceHybrid
	}
	if cfg.GroundingPolicy == "" {
		// Default to soft down-weighting: hard-dropping the least mutually
		// consistent response regresses quality on contested factual items
		// (see bench/grounded_fusion/FINDINGS.md). Set policy=filter for the
		// prior drop behavior.
		cfg.GroundingPolicy = config.FusionGroundingPolicyWeight
	}
	if cfg.GroundingMinKeep <= 0 {
		cfg.GroundingMinKeep = 1
	}
	if cfg.GroundingNLIContradictionPenalty <= 0 {
		cfg.GroundingNLIContradictionPenalty = 1.0
	}
	if strings.TrimSpace(cfg.GroundingOnError) == "" {
		cfg.GroundingOnError = cfg.OnError
	}
}

func validateFusionExecutionConfig(cfg fusionExecutionConfig) error {
	if cfg.MinSuccessfulResponses > len(cfg.AnalysisModels) {
		return fmt.Errorf(
			"fusion min_successful_responses=%d exceeds panel size %d",
			cfg.MinSuccessfulResponses,
			len(cfg.AnalysisModels),
		)
	}
	switch cfg.AnalysisMode {
	case config.FusionAnalysisModeSeparate, config.FusionAnalysisModeOneCall, config.FusionAnalysisModeNone:
	default:
		return fmt.Errorf(
			"fusion analysis_mode must be %q, %q, or %q, got %q",
			config.FusionAnalysisModeSeparate,
			config.FusionAnalysisModeOneCall,
			config.FusionAnalysisModeNone,
			cfg.AnalysisMode,
		)
	}
	if cfg.AnalysisMode != config.FusionAnalysisModeSeparate && strings.TrimSpace(cfg.AnalysisTemplate) != "" {
		return fmt.Errorf("fusion analysis_template requires analysis_mode=%q", config.FusionAnalysisModeSeparate)
	}
	switch cfg.OnError {
	case config.FusionOnErrorSkip, config.FusionOnErrorFail:
		return nil
	default:
		return fmt.Errorf("fusion on_error must be %q or %q, got %q", config.FusionOnErrorSkip, config.FusionOnErrorFail, cfg.OnError)
	}
}

func mergeFusionAlgorithmConfig(dst *fusionExecutionConfig, src *config.FusionAlgorithmConfig) {
	mergeFusionModels(dst, src.Model, src.AnalysisModels)
	if src.AnalysisMode != "" {
		dst.AnalysisMode = src.AnalysisMode
	}
	mergeFusionAnalysisOverrides(dst, src.AnalysisOverrides)
	mergeFusionLimits(dst, src.MaxConcurrent, src.MaxCompletionTokens, src.RoundTimeoutSeconds, src.MinSuccessfulResponses)
	mergeFusionControls(dst, src.Temperature, src.IncludeAnalysis, src.IncludeIntermediateResponses, src.OnError)
	mergeFusionPrompts(dst, src.AnalysisTemplate, src.SynthesisTemplate, src.JudgePromptVersion)
	mergeFusionQuorumFailure(dst, src.QuorumFailurePolicy, src.QuorumFallbackTarget)
	mergeFusionGroundingConfig(dst, src.Grounding)
}

// mergeFusionQuorumFailure copies the recipe-owned below-quorum policy. There is
// deliberately no request-level counterpart: request input must not weaken the
// operator's configured quality boundary.
func mergeFusionQuorumFailure(
	dst *fusionExecutionConfig,
	policy config.FusionQuorumFailurePolicy,
	fallbackTarget string,
) {
	if policy != "" {
		dst.QuorumFailurePolicy = policy
	}
	if trimmed := strings.TrimSpace(fallbackTarget); trimmed != "" {
		dst.QuorumFallbackTarget = trimmed
	}
}

func mergeFusionModels(dst *fusionExecutionConfig, judgeModel string, analysisModels []string) {
	if judgeModel != "" {
		dst.Model = judgeModel
	}
	if len(analysisModels) > 0 {
		dst.AnalysisModels = append([]string(nil), analysisModels...)
	}
}

func mergeFusionLimits(
	dst *fusionExecutionConfig,
	maxConcurrent int,
	maxCompletionTokens int,
	roundTimeoutSeconds int,
	minSuccessfulResponses int,
) {
	if maxConcurrent > 0 {
		dst.MaxConcurrent = maxConcurrent
	}
	if maxCompletionTokens > 0 {
		dst.MaxCompletionTokens = maxCompletionTokens
	}
	if roundTimeoutSeconds > 0 {
		dst.RoundTimeoutSeconds = roundTimeoutSeconds
	}
	if minSuccessfulResponses > 0 {
		dst.MinSuccessfulResponses = minSuccessfulResponses
	}
}

func mergeFusionControls(
	dst *fusionExecutionConfig,
	temperature *float64,
	includeAnalysis *bool,
	includeIntermediateResponses *bool,
	onError string,
) {
	if temperature != nil {
		dst.Temperature = temperature
	}
	mergeFusionTraceVisibility(dst, includeAnalysis, includeIntermediateResponses)
	if onError != "" {
		dst.OnError = onError
	}
}

func mergeFusionTraceVisibility(
	dst *fusionExecutionConfig,
	includeAnalysis *bool,
	includeIntermediateResponses *bool,
) {
	if includeAnalysis != nil {
		dst.IncludeAnalysis = *includeAnalysis
	}
	if includeIntermediateResponses != nil {
		dst.IncludeIntermediateResponses = *includeIntermediateResponses
	}
}

func mergeFusionPrompts(
	dst *fusionExecutionConfig,
	analysisTemplate string,
	synthesisTemplate string,
	judgePromptVersion string,
) {
	if analysisTemplate != "" {
		dst.AnalysisTemplate = analysisTemplate
	}
	if synthesisTemplate != "" {
		dst.SynthesisTemplate = synthesisTemplate
	}
	if judgePromptVersion != "" {
		dst.JudgePromptVersion = judgePromptVersion
	}
}

func mergeFusionGroundingConfig(dst *fusionExecutionConfig, src *config.FusionGroundingConfig) {
	if src == nil {
		return
	}
	dst.GroundingEnabled = src.Enabled
	dst.GroundingReference = src.Reference
	dst.GroundingPolicy = src.Policy
	dst.GroundingMinScore = src.MinScore
	dst.GroundingMinKeep = src.MinKeep
	dst.GroundingNLIContradictionPenalty = src.NLIContradictionPenalty
	dst.GroundingOnError = src.OnError
}

func mergeFusionRequestConfig(dst *fusionExecutionConfig, src *config.FusionRequestConfig) {
	mergeFusionModels(dst, src.Model, src.AnalysisModels)
	mergeFusionAnalysisOverrides(dst, src.AnalysisOverrides)
	mergeFusionLimits(dst, src.MaxConcurrent, src.MaxCompletionTokens, src.RoundTimeoutSeconds, src.MinSuccessfulResponses)
	mergeFusionControls(dst, src.Temperature, src.IncludeAnalysis, src.IncludeIntermediateResponses, src.OnError)
	mergeFusionPrompts(dst, src.AnalysisTemplate, src.SynthesisTemplate, src.JudgePromptVersion)
	mergeFusionGroundingConfig(dst, src.Grounding)
}

func fusionRequestEnabled(req *config.FusionRequestConfig) bool {
	return req.Enabled == nil || *req.Enabled
}

func modelRefsToNames(modelRefs []config.ModelRef) []string {
	names := make([]string, 0, len(modelRefs))
	for _, ref := range modelRefs {
		if ref.LoRAName != "" {
			names = append(names, ref.LoRAName)
			continue
		}
		names = append(names, ref.Model)
	}
	return names
}

func normalizeModelNames(names []string) []string {
	seen := make(map[string]bool, len(names))
	result := make([]string, 0, len(names))
	for _, name := range names {
		trimmed := strings.TrimSpace(name)
		if trimmed == "" || seen[trimmed] {
			continue
		}
		seen[trimmed] = true
		result = append(result, trimmed)
	}
	return result
}

// mergeFusionAnalysisOverrides accumulates sparse per-model overrides
// field-wise, preserving an existing sampling field when an incoming entry
// omits it.
func mergeFusionAnalysisOverrides(dst *fusionExecutionConfig, overrides []config.FusionModelOverride) {
	if len(overrides) == 0 {
		return
	}
	if dst.AnalysisOverrides == nil {
		dst.AnalysisOverrides = make(map[string]config.FusionModelOverride, len(overrides))
	}
	for _, override := range overrides {
		name := strings.TrimSpace(override.Model)
		if name == "" {
			continue
		}
		merged := dst.AnalysisOverrides[name]
		merged.Model = name
		if override.Temperature != nil {
			merged.Temperature = override.Temperature
		}
		if override.MaxCompletionTokens > 0 {
			merged.MaxCompletionTokens = override.MaxCompletionTokens
		}
		dst.AnalysisOverrides[name] = merged
	}
}

func normalizeFusionAnalysisOverrides(
	analysisModels []string,
	overrides map[string]config.FusionModelOverride,
) map[string]config.FusionModelOverride {
	if len(overrides) == 0 || len(analysisModels) == 0 {
		return nil
	}
	allowed := make(map[string]bool, len(analysisModels))
	for _, model := range analysisModels {
		allowed[model] = true
	}
	filtered := make(map[string]config.FusionModelOverride, len(overrides))
	for model, override := range overrides {
		if !allowed[model] {
			continue
		}
		filtered[model] = override
	}
	if len(filtered) == 0 {
		return nil
	}
	return filtered
}
