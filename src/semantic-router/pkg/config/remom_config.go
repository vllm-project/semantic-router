package config

import (
	"fmt"
	"strings"
)

const (
	DefaultReMoMModelName = "vllm-sr/remom"

	ReMoMDistributionWeighted   = "weighted"
	ReMoMDistributionEqual      = "equal"
	ReMoMDistributionRoundRobin = "round_robin"
	ReMoMDistributionFirstOnly  = "first_only"

	ReMoMCompactionFull        = "full"
	ReMoMCompactionLastNTokens = "last_n_tokens"

	ReMoMOnErrorSkip = "skip"
	ReMoMOnErrorFail = "fail"
)

// ReMoMRuntimeConfig registers direct ReMoM model slugs. ReMoM breadth,
// compaction, and synthesis policy live on routing decisions.
type ReMoMRuntimeConfig struct {
	ModelNames []string `yaml:"model_names,omitempty" json:"model_names,omitempty"`
}

func DefaultReMoMModelNames() []string {
	return []string{DefaultReMoMModelName}
}

func (c ReMoMRuntimeConfig) EffectiveModelNames() []string {
	if len(c.ModelNames) > 0 {
		return normalizeReMoMModelNames(c.ModelNames)
	}
	return DefaultReMoMModelNames()
}

func (c *RouterConfig) ExposedReMoMModelNames() []string {
	if c == nil || !c.Looper.IsEnabled() {
		return nil
	}
	if !c.HasReMoMDecision() {
		return nil
	}
	return c.Looper.ReMoM.EffectiveModelNames()
}

func normalizeReMoMModelNames(names []string) []string {
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

func (c *RouterConfig) IsReMoMModelName(modelName string) bool {
	if c == nil {
		return false
	}
	normalized := strings.TrimSpace(modelName)
	if normalized == "" {
		return false
	}
	for _, candidate := range c.Looper.ReMoM.EffectiveModelNames() {
		if normalized == candidate {
			return true
		}
	}
	return false
}

func (c *RouterConfig) HasReMoMDecision() bool {
	if c == nil {
		return false
	}
	for _, decision := range c.Decisions {
		if decision.Algorithm != nil && decision.Algorithm.Type == "remom" {
			return true
		}
	}
	return false
}

func ValidateReMoMRuntimeConfig(cfg ReMoMRuntimeConfig) error {
	for i, name := range cfg.ModelNames {
		if strings.TrimSpace(name) == "" {
			return fmt.Errorf("model_names[%d] cannot be empty", i)
		}
	}
	return nil
}

func ValidateReMoMAlgorithmConfig(cfg *ReMoMAlgorithmConfig) error {
	if cfg == nil {
		return nil
	}
	if err := validateReMoMBreadthSchedule(cfg.BreadthSchedule); err != nil {
		return err
	}
	if err := validateReMoMDistribution(cfg.ModelDistribution); err != nil {
		return err
	}
	if cfg.Temperature < 0 {
		return fmt.Errorf("temperature must be >= 0 when set")
	}
	if err := validateReMoMCompactionStrategy(cfg.CompactionStrategy); err != nil {
		return err
	}
	if err := validateReMoMPositiveControls(cfg); err != nil {
		return err
	}
	return validateReMoMOnError(cfg.OnError)
}

func validateReMoMBreadthSchedule(schedule []int) error {
	if len(schedule) == 0 {
		return fmt.Errorf("breadth_schedule cannot be empty")
	}
	for i, breadth := range schedule {
		if breadth <= 0 {
			return fmt.Errorf("breadth_schedule[%d] must be >= 1", i)
		}
	}
	return nil
}

func validateReMoMPositiveControls(cfg *ReMoMAlgorithmConfig) error {
	if cfg.CompactionTokens < 0 {
		return fmt.Errorf("compaction_tokens must be >= 0")
	}
	if cfg.MaxConcurrent < 0 {
		return fmt.Errorf("max_concurrent must be >= 1 when set")
	}
	if cfg.RoundTimeoutSeconds < 0 {
		return fmt.Errorf("round_timeout_seconds must be >= 1 when set")
	}
	if cfg.MinSuccessfulResponses < 0 {
		return fmt.Errorf("min_successful_responses must be >= 1 when set")
	}
	if cfg.MaxResponsesPerRound < 0 {
		return fmt.Errorf("max_responses_per_round must be >= 1 when set")
	}
	return nil
}

func ValidateReMoMModelRefs(cfg *ReMoMAlgorithmConfig, modelRefs []ModelRef) error {
	if cfg == nil || strings.TrimSpace(cfg.SynthesisModel) == "" {
		return nil
	}
	for _, ref := range modelRefs {
		if ref.Model == cfg.SynthesisModel {
			return nil
		}
	}
	return fmt.Errorf("synthesis_model %q must reference a decision modelRef", cfg.SynthesisModel)
}

func validateReMoMDistribution(distribution string) error {
	switch strings.TrimSpace(distribution) {
	case "", ReMoMDistributionWeighted, ReMoMDistributionEqual, ReMoMDistributionRoundRobin, ReMoMDistributionFirstOnly:
		return nil
	default:
		return fmt.Errorf(
			"model_distribution must be one of %q, %q, %q, or %q, got %q",
			ReMoMDistributionWeighted,
			ReMoMDistributionEqual,
			ReMoMDistributionRoundRobin,
			ReMoMDistributionFirstOnly,
			distribution,
		)
	}
}

func validateReMoMCompactionStrategy(strategy string) error {
	switch strings.TrimSpace(strategy) {
	case "", ReMoMCompactionFull, ReMoMCompactionLastNTokens:
		return nil
	default:
		return fmt.Errorf(
			"compaction_strategy must be one of %q or %q, got %q",
			ReMoMCompactionFull,
			ReMoMCompactionLastNTokens,
			strategy,
		)
	}
}

func validateReMoMOnError(onError string) error {
	switch strings.TrimSpace(onError) {
	case "", ReMoMOnErrorSkip, ReMoMOnErrorFail:
		return nil
	default:
		return fmt.Errorf("on_error must be one of %q or %q, got %q", ReMoMOnErrorSkip, ReMoMOnErrorFail, onError)
	}
}
