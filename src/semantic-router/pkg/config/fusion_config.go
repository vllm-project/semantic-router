package config

import (
	"fmt"
	"strings"
)

const (
	DefaultFusionModelName          = "vllm-sr/fusion"
	OpenRouterFusionModelAlias      = "openrouter/fusion"
	DefaultFusionJudgePromptVersion = "fusion-v1"
	FusionOnErrorSkip               = "skip"
	FusionOnErrorFail               = "fail"
)

// FusionAlgorithmConfig configures Fusion-style panel execution for
// decision.algorithm.type=fusion. Model is the judge/calling model; analysis
// models come from analysis_models when set, otherwise from decision.modelRefs.
type FusionAlgorithmConfig struct {
	Model                        string   `yaml:"model,omitempty" json:"model,omitempty"`
	AnalysisModels               []string `yaml:"analysis_models,omitempty" json:"analysis_models,omitempty"`
	MaxConcurrent                int      `yaml:"max_concurrent,omitempty" json:"max_concurrent,omitempty"`
	MaxCompletionTokens          int      `yaml:"max_completion_tokens,omitempty" json:"max_completion_tokens,omitempty"`
	Temperature                  *float64 `yaml:"temperature,omitempty" json:"temperature,omitempty"`
	IncludeAnalysis              *bool    `yaml:"include_analysis,omitempty" json:"include_analysis,omitempty"`
	OnError                      string   `yaml:"on_error,omitempty" json:"on_error,omitempty"`
	AnalysisTemplate             string   `yaml:"analysis_template,omitempty" json:"analysis_template,omitempty"`
	SynthesisTemplate            string   `yaml:"synthesis_template,omitempty" json:"synthesis_template,omitempty"`
	JudgePromptVersion           string   `yaml:"judge_prompt_version,omitempty" json:"judge_prompt_version,omitempty"`
	IncludeIntermediateResponses *bool    `yaml:"include_intermediate_responses,omitempty" json:"include_intermediate_responses,omitempty"`
}

// FusionRuntimeConfig registers direct Fusion model slugs. The panel and judge
// policy live on routing decisions, not in global runtime config.
type FusionRuntimeConfig struct {
	ModelNames []string `yaml:"model_names,omitempty" json:"model_names,omitempty"`
}

// FusionRequestConfig is the request-level OpenAI-compatible extension parsed
// from plugins[].id=fusion. It intentionally uses JSON tags first because it is
// not a decision plugin config surface.
type FusionRequestConfig struct {
	ID                  string   `json:"id" yaml:"id"`
	Enabled             *bool    `json:"enabled,omitempty" yaml:"enabled,omitempty"`
	Model               string   `json:"model,omitempty" yaml:"model,omitempty"`
	AnalysisModels      []string `json:"analysis_models,omitempty" yaml:"analysis_models,omitempty"`
	MaxConcurrent       int      `json:"max_concurrent,omitempty" yaml:"max_concurrent,omitempty"`
	MaxCompletionTokens int      `json:"max_completion_tokens,omitempty" yaml:"max_completion_tokens,omitempty"`
	Temperature         *float64 `json:"temperature,omitempty" yaml:"temperature,omitempty"`
	OnError             string   `json:"on_error,omitempty" yaml:"on_error,omitempty"`
}

func DefaultFusionModelNames() []string {
	return []string{DefaultFusionModelName}
}

func (c FusionRuntimeConfig) EffectiveModelNames() []string {
	if len(c.ModelNames) > 0 {
		return normalizeFusionModelNames(c.ModelNames)
	}
	return DefaultFusionModelNames()
}

func normalizeFusionModelNames(names []string) []string {
	seen := make(map[string]bool, len(names))
	result := make([]string, 0, len(names))
	for _, name := range names {
		normalized := strings.TrimSpace(name)
		if normalized == "" || seen[normalized] {
			continue
		}
		seen[normalized] = true
		result = append(result, normalized)
	}
	return result
}

func (c *RouterConfig) IsFusionModelName(modelName string) bool {
	if c == nil {
		return false
	}
	normalized := strings.TrimSpace(modelName)
	if normalized == "" {
		return false
	}
	for _, candidate := range c.Looper.Fusion.EffectiveModelNames() {
		if normalized == candidate {
			return true
		}
	}
	return false
}

func ValidateFusionAlgorithmConfig(cfg *FusionAlgorithmConfig) error {
	if cfg == nil {
		return nil
	}
	if err := validateFusionOnError(cfg.OnError); err != nil {
		return err
	}
	if cfg.MaxConcurrent < 0 {
		return fmt.Errorf("max_concurrent must be >= 1 when set")
	}
	if cfg.MaxCompletionTokens < 0 {
		return fmt.Errorf("max_completion_tokens must be >= 1 when set")
	}
	if cfg.Temperature != nil && *cfg.Temperature < 0 {
		return fmt.Errorf("temperature must be >= 0 when set")
	}
	return nil
}

func ValidateFusionRuntimeConfig(cfg FusionRuntimeConfig) error {
	for i, name := range cfg.ModelNames {
		if strings.TrimSpace(name) == "" {
			return fmt.Errorf("model_names[%d] cannot be empty", i)
		}
	}
	return nil
}

func (c *FusionRequestConfig) Validate() error {
	if c == nil {
		return nil
	}
	if err := validateFusionOnError(c.OnError); err != nil {
		return err
	}
	if c.MaxConcurrent < 0 {
		return fmt.Errorf("max_concurrent must be >= 1 when set")
	}
	if c.MaxCompletionTokens < 0 {
		return fmt.Errorf("max_completion_tokens must be >= 1 when set")
	}
	if c.Temperature != nil && *c.Temperature < 0 {
		return fmt.Errorf("temperature must be >= 0 when set")
	}
	return nil
}

func validateFusionOnError(onError string) error {
	if strings.TrimSpace(onError) == "" {
		return nil
	}
	switch onError {
	case FusionOnErrorSkip, FusionOnErrorFail:
		return nil
	default:
		return fmt.Errorf("on_error must be one of %q or %q, got %q", FusionOnErrorSkip, FusionOnErrorFail, onError)
	}
}
