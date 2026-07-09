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

	// Grounding reference modes select what panel responses are scored against.
	FusionGroundingReferenceHybrid  = "hybrid"  // detector against context if present, else cross-model NLI
	FusionGroundingReferenceContext = "context" // detector against provided RAG/tool context only
	FusionGroundingReferencePanel   = "panel"   // cross-model NLI only (panel as its own reference)

	// Grounding policy modes select how the groundedness scores are used.
	//   - weight:   keep every response, tell the judge to weight each panel
	//               answer by its groundedness score (soft down-weighting).
	//   - annotate: keep every response, surface groundedness notes to the judge
	//               without a weighting instruction.
	//   - filter:   hard-drop responses below min_score (keep min_keep).
	// The default is weight. Hard-dropping the least mutually-consistent response
	// regresses quality on contested factual items (it deletes the correct
	// dissenter); see bench/grounded_fusion/FINDINGS.md.
	FusionGroundingPolicyWeight   = "weight"
	FusionGroundingPolicyAnnotate = "annotate"
	FusionGroundingPolicyFilter   = "filter"
)

// FusionAlgorithmConfig configures Fusion-style panel execution for
// decision.algorithm.type=fusion. Model is the judge/calling model; analysis
// models come from analysis_models when set, otherwise from decision.modelRefs.
type FusionAlgorithmConfig struct {
	Model                        string                 `yaml:"model,omitempty" json:"model,omitempty"`
	AnalysisModels               []string               `yaml:"analysis_models,omitempty" json:"analysis_models,omitempty"`
	MaxConcurrent                int                    `yaml:"max_concurrent,omitempty" json:"max_concurrent,omitempty"`
	MaxCompletionTokens          int                    `yaml:"max_completion_tokens,omitempty" json:"max_completion_tokens,omitempty"`
	RoundTimeoutSeconds          int                    `yaml:"round_timeout_seconds,omitempty" json:"round_timeout_seconds,omitempty"`
	MinSuccessfulResponses       int                    `yaml:"min_successful_responses,omitempty" json:"min_successful_responses,omitempty"`
	Temperature                  *float64               `yaml:"temperature,omitempty" json:"temperature,omitempty"`
	IncludeAnalysis              *bool                  `yaml:"include_analysis,omitempty" json:"include_analysis,omitempty"`
	OnError                      string                 `yaml:"on_error,omitempty" json:"on_error,omitempty"`
	AnalysisTemplate             string                 `yaml:"analysis_template,omitempty" json:"analysis_template,omitempty"`
	SynthesisTemplate            string                 `yaml:"synthesis_template,omitempty" json:"synthesis_template,omitempty"`
	JudgePromptVersion           string                 `yaml:"judge_prompt_version,omitempty" json:"judge_prompt_version,omitempty"`
	IncludeIntermediateResponses *bool                  `yaml:"include_intermediate_responses,omitempty" json:"include_intermediate_responses,omitempty"`
	Grounding                    *FusionGroundingConfig `yaml:"grounding,omitempty" json:"grounding,omitempty"`
}

// FusionGroundingConfig configures the optional grounding stage that scores each
// panel response for faithfulness before the judge synthesizes. When nil or
// disabled, Fusion behaves exactly as without grounding. Grounding makes no extra
// LLM calls: it uses local encoder models (hallucination detector + NLI).
type FusionGroundingConfig struct {
	Enabled                 bool    `yaml:"enabled,omitempty" json:"enabled,omitempty"`
	Reference               string  `yaml:"reference,omitempty" json:"reference,omitempty"`
	Policy                  string  `yaml:"policy,omitempty" json:"policy,omitempty"`
	MinScore                float64 `yaml:"min_score,omitempty" json:"min_score,omitempty"`
	MinKeep                 int     `yaml:"min_keep,omitempty" json:"min_keep,omitempty"`
	NLIContradictionPenalty float64 `yaml:"nli_contradiction_penalty,omitempty" json:"nli_contradiction_penalty,omitempty"`
	OnError                 string  `yaml:"on_error,omitempty" json:"on_error,omitempty"`
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
	ID                           string                 `json:"id" yaml:"id"`
	Enabled                      *bool                  `json:"enabled,omitempty" yaml:"enabled,omitempty"`
	Model                        string                 `json:"model,omitempty" yaml:"model,omitempty"`
	AnalysisModels               []string               `json:"analysis_models,omitempty" yaml:"analysis_models,omitempty"`
	MaxConcurrent                int                    `json:"max_concurrent,omitempty" yaml:"max_concurrent,omitempty"`
	MaxCompletionTokens          int                    `json:"max_completion_tokens,omitempty" yaml:"max_completion_tokens,omitempty"`
	RoundTimeoutSeconds          int                    `json:"round_timeout_seconds,omitempty" yaml:"round_timeout_seconds,omitempty"`
	MinSuccessfulResponses       int                    `json:"min_successful_responses,omitempty" yaml:"min_successful_responses,omitempty"`
	Temperature                  *float64               `json:"temperature,omitempty" yaml:"temperature,omitempty"`
	IncludeAnalysis              *bool                  `json:"include_analysis,omitempty" yaml:"include_analysis,omitempty"`
	IncludeIntermediateResponses *bool                  `json:"include_intermediate_responses,omitempty" yaml:"include_intermediate_responses,omitempty"`
	OnError                      string                 `json:"on_error,omitempty" yaml:"on_error,omitempty"`
	AnalysisTemplate             string                 `json:"analysis_template,omitempty" yaml:"analysis_template,omitempty"`
	SynthesisTemplate            string                 `json:"synthesis_template,omitempty" yaml:"synthesis_template,omitempty"`
	JudgePromptVersion           string                 `json:"judge_prompt_version,omitempty" yaml:"judge_prompt_version,omitempty"`
	Grounding                    *FusionGroundingConfig `json:"grounding,omitempty" yaml:"grounding,omitempty"`
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

func (c *RouterConfig) ExposedFusionModelNames() []string {
	if c == nil || !c.Looper.IsEnabled() {
		return nil
	}
	if len(c.Looper.Fusion.ModelNames) == 0 && !c.HasFusionDecision() {
		return nil
	}
	return c.Looper.Fusion.EffectiveModelNames()
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

func (c *RouterConfig) HasFusionDecision() bool {
	if c == nil {
		return false
	}
	for _, decision := range c.Decisions {
		if decision.Algorithm != nil && decision.Algorithm.Type == "fusion" {
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
	if cfg.RoundTimeoutSeconds < 0 {
		return fmt.Errorf("round_timeout_seconds must be >= 1 when set")
	}
	if cfg.MinSuccessfulResponses < 0 {
		return fmt.Errorf("min_successful_responses must be >= 1 when set")
	}
	if cfg.Temperature != nil && *cfg.Temperature < 0 {
		return fmt.Errorf("temperature must be >= 0 when set")
	}
	if err := ValidateFusionGroundingConfig(cfg.Grounding); err != nil {
		return err
	}
	return nil
}

// ValidateFusionGroundingConfig validates the grounding block. Bounds are kept
// identical to the Python CLI mirror (see src/vllm-sr/cli/algorithms.py).
func ValidateFusionGroundingConfig(cfg *FusionGroundingConfig) error {
	if cfg == nil {
		return nil
	}
	switch strings.TrimSpace(cfg.Reference) {
	case "", FusionGroundingReferenceHybrid, FusionGroundingReferenceContext, FusionGroundingReferencePanel:
	default:
		return fmt.Errorf("grounding.reference must be one of %q, %q or %q, got %q",
			FusionGroundingReferenceHybrid, FusionGroundingReferenceContext, FusionGroundingReferencePanel, cfg.Reference)
	}
	switch strings.TrimSpace(cfg.Policy) {
	case "", FusionGroundingPolicyWeight, FusionGroundingPolicyAnnotate, FusionGroundingPolicyFilter:
	default:
		return fmt.Errorf("grounding.policy must be one of %q, %q or %q, got %q",
			FusionGroundingPolicyWeight, FusionGroundingPolicyAnnotate, FusionGroundingPolicyFilter, cfg.Policy)
	}
	if cfg.MinScore < 0 || cfg.MinScore > 1 {
		return fmt.Errorf("grounding.min_score must be between 0 and 1")
	}
	if cfg.MinKeep < 0 {
		return fmt.Errorf("grounding.min_keep must be >= 0")
	}
	if cfg.NLIContradictionPenalty < 0 {
		return fmt.Errorf("grounding.nli_contradiction_penalty must be >= 0")
	}
	return validateFusionOnError(cfg.OnError)
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
	if c.RoundTimeoutSeconds < 0 {
		return fmt.Errorf("round_timeout_seconds must be >= 1 when set")
	}
	if c.MinSuccessfulResponses < 0 {
		return fmt.Errorf("min_successful_responses must be >= 1 when set")
	}
	if c.Temperature != nil && *c.Temperature < 0 {
		return fmt.Errorf("temperature must be >= 0 when set")
	}
	if err := ValidateFusionGroundingConfig(c.Grounding); err != nil {
		return err
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
