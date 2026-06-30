package looper

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (l *FusionLooper) resolveFusionExecutionConfig(req *Request) fusionExecutionConfig {
	cfg := fusionExecutionConfig{
		IncludeAnalysis:              true,
		IncludeIntermediateResponses: true,
	}

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
		mergeFusionRequestConfig(&cfg, req.Fusion)
	}
	return normalizeFusionExecutionConfig(cfg)
}

func normalizeFusionExecutionConfig(cfg fusionExecutionConfig) fusionExecutionConfig {
	cfg.OnError = strings.TrimSpace(cfg.OnError)
	if cfg.OnError == "" {
		cfg.OnError = config.FusionOnErrorSkip
	}
	if cfg.JudgePromptVersion == "" {
		cfg.JudgePromptVersion = config.DefaultFusionJudgePromptVersion
	}
	cfg.AnalysisModels = normalizeModelNames(cfg.AnalysisModels)
	if cfg.MaxConcurrent <= 0 || cfg.MaxConcurrent > len(cfg.AnalysisModels) {
		cfg.MaxConcurrent = len(cfg.AnalysisModels)
	}
	if cfg.MinSuccessfulResponses <= 0 || cfg.MinSuccessfulResponses > len(cfg.AnalysisModels) {
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
	switch cfg.OnError {
	case config.FusionOnErrorSkip, config.FusionOnErrorFail:
		return nil
	default:
		return fmt.Errorf("fusion on_error must be %q or %q, got %q", config.FusionOnErrorSkip, config.FusionOnErrorFail, cfg.OnError)
	}
}

func mergeFusionAlgorithmConfig(dst *fusionExecutionConfig, src *config.FusionAlgorithmConfig) {
	mergeFusionModels(dst, src.Model, src.AnalysisModels)
	mergeFusionLimits(dst, src.MaxConcurrent, src.MaxCompletionTokens, src.RoundTimeoutSeconds, src.MinSuccessfulResponses)
	mergeFusionControls(dst, src.Temperature, src.IncludeAnalysis, src.IncludeIntermediateResponses, src.OnError)
	mergeFusionPrompts(dst, src.AnalysisTemplate, src.SynthesisTemplate, src.JudgePromptVersion)
	mergeFusionGroundingConfig(dst, src.Grounding)
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
	if includeAnalysis != nil {
		dst.IncludeAnalysis = *includeAnalysis
	}
	if includeIntermediateResponses != nil {
		dst.IncludeIntermediateResponses = *includeIntermediateResponses
	}
	if onError != "" {
		dst.OnError = onError
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
