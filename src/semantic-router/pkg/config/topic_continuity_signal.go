package config

import (
	"fmt"
	"math"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/topiccontinuity"
)

// SignalTypeTopicContinuity is a request-stage evidence source for context
// policy. It is not decision-referenceable: the selected recipe evaluates
// every declared rule before the context transformation stage, and consumers
// read typed results from the request context.
const SignalTypeTopicContinuity = "topic_continuity"

// TopicContinuityMaxRules bounds the rules one recipe may declare, which
// bounds per-request evaluation work.
const TopicContinuityMaxRules = 8

const (
	defaultTopicContinuityThreshold    = 0.35
	defaultTopicChangeThreshold        = 0.08
	defaultTopicContinuityPriorTurns   = 8
	defaultTopicContinuityTurnBytes    = 16384
	defaultTopicContinuityIncludeAsstn = true
)

// TopicContinuityRule declares one topic-continuity evaluation. Every field
// except name is optional; pointer fields distinguish an omitted value from
// an explicit false or zero.
type TopicContinuityRule struct {
	Name             string                         `yaml:"name"`
	Description      string                         `yaml:"description,omitempty"`
	IncludeAssistant *bool                          `yaml:"include_assistant,omitempty"`
	Thresholds       *TopicContinuityThresholds     `yaml:"thresholds,omitempty"`
	Limits           *TopicContinuityEvidenceLimits `yaml:"limits,omitempty"`
}

// TopicContinuityThresholds sets the continuation and change score bounds.
// The gap between them is the unknown band.
type TopicContinuityThresholds struct {
	Continuation *float64 `yaml:"continuation,omitempty"`
	Change       *float64 `yaml:"change,omitempty"`
}

// TopicContinuityEvidenceLimits bounds the history one rule reads. Zero means
// omitted; zero is outside every valid range.
type TopicContinuityEvidenceLimits struct {
	MaxPriorTurns int `yaml:"max_prior_turns,omitempty"`
	MaxTurnBytes  int `yaml:"max_turn_bytes,omitempty"`
	MaxInputBytes int `yaml:"max_input_bytes,omitempty"`
}

// EffectiveIncludeAssistant applies the default.
func (r TopicContinuityRule) EffectiveIncludeAssistant() bool {
	if r.IncludeAssistant == nil {
		return defaultTopicContinuityIncludeAsstn
	}
	return *r.IncludeAssistant
}

// EffectiveThresholds returns (continuation, change) with defaults applied.
func (r TopicContinuityRule) EffectiveThresholds() (float64, float64) {
	continuation, change := defaultTopicContinuityThreshold, defaultTopicChangeThreshold
	if r.Thresholds != nil {
		if r.Thresholds.Continuation != nil {
			continuation = *r.Thresholds.Continuation
		}
		if r.Thresholds.Change != nil {
			change = *r.Thresholds.Change
		}
	}
	return continuation, change
}

// EffectiveLimits applies defaults. An omitted max_input_bytes is derived as
// (max_prior_turns + 1) * max_turn_bytes from the effective turn values;
// callers validate the ranges before relying on the derived value.
func (r TopicContinuityRule) EffectiveLimits() topiccontinuity.Limits {
	limits := topiccontinuity.Limits{
		MaxPriorTurns: defaultTopicContinuityPriorTurns,
		MaxTurnBytes:  defaultTopicContinuityTurnBytes,
	}
	if r.Limits != nil {
		if r.Limits.MaxPriorTurns != 0 {
			limits.MaxPriorTurns = r.Limits.MaxPriorTurns
		}
		if r.Limits.MaxTurnBytes != 0 {
			limits.MaxTurnBytes = r.Limits.MaxTurnBytes
		}
		limits.MaxInputBytes = r.Limits.MaxInputBytes
	}
	if limits.MaxInputBytes == 0 && validTurnRange(limits) {
		limits.MaxInputBytes = (limits.MaxPriorTurns + 1) * limits.MaxTurnBytes
	}
	return limits
}

func (r TopicContinuityRule) inputBytesDerived() bool {
	return r.Limits == nil || r.Limits.MaxInputBytes == 0
}

func validTurnRange(limits topiccontinuity.Limits) bool {
	return limits.MaxPriorTurns >= topiccontinuity.MinPriorTurns &&
		limits.MaxPriorTurns <= topiccontinuity.MaxPriorTurns &&
		limits.MaxTurnBytes >= topiccontinuity.MinTurnBytes &&
		limits.MaxTurnBytes <= topiccontinuity.MaxTurnBytes
}

// EvalConfig translates the validated rule into the evaluator's pure config.
// This is the only translation point between configuration and evaluation.
func (r TopicContinuityRule) EvalConfig() topiccontinuity.EvalConfig {
	continuation, change := r.EffectiveThresholds()
	return topiccontinuity.EvalConfig{
		Name: r.Name,
		Policy: topiccontinuity.HistoryPolicy{
			Limits:           r.EffectiveLimits(),
			IncludeAssistant: r.EffectiveIncludeAssistant(),
		},
		Continuation: continuation,
		Change:       change,
	}
}

func validateTopicContinuityContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	if len(cfg.TopicContinuityRules) > TopicContinuityMaxRules {
		return fmt.Errorf("routing.signals.topic_continuity: at most %d rules are allowed, got %d",
			TopicContinuityMaxRules, len(cfg.TopicContinuityRules))
	}
	seen := make(map[string]struct{}, len(cfg.TopicContinuityRules))
	for i, rule := range cfg.TopicContinuityRules {
		if err := ValidateTopicContinuityRuleContract(rule); err != nil {
			return fmt.Errorf("routing.signals.topic_continuity[%d]: %w", i, err)
		}
		if _, exists := seen[rule.Name]; exists {
			return fmt.Errorf("routing.signals.topic_continuity[%d]: duplicate name %q", i, rule.Name)
		}
		seen[rule.Name] = struct{}{}
	}
	return nil
}

// ValidateTopicContinuityRuleContract validates one rule on its effective
// values. Exported so the DSL compiler shares the exact same contract.
func ValidateTopicContinuityRuleContract(rule TopicContinuityRule) error {
	name := strings.TrimSpace(rule.Name)
	if name == "" {
		return fmt.Errorf("name is required")
	}
	if name != rule.Name {
		return fmt.Errorf("name must not contain surrounding whitespace")
	}
	continuation, change := rule.EffectiveThresholds()
	finite := func(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
	if !finite(continuation) || !finite(change) || change < 0 || change >= continuation || continuation >= 1 {
		return fmt.Errorf("thresholds must satisfy 0 <= change < continuation < 1, got change=%v continuation=%v",
			change, continuation)
	}
	return validateTopicContinuityLimits(rule)
}

// validateTopicContinuityLimits range-checks each limit before the derived
// total is computed, so the derivation never relies on overflow behavior.
func validateTopicContinuityLimits(rule TopicContinuityRule) error {
	if rule.Limits != nil {
		raw := rule.Limits
		if raw.MaxPriorTurns < 0 || raw.MaxTurnBytes < 0 || raw.MaxInputBytes < 0 {
			return fmt.Errorf("limits must not be negative")
		}
	}
	limits := rule.EffectiveLimits()
	if limits.MaxPriorTurns < topiccontinuity.MinPriorTurns || limits.MaxPriorTurns > topiccontinuity.MaxPriorTurns {
		return fmt.Errorf("limits.max_prior_turns must be in [%d, %d], got %d",
			topiccontinuity.MinPriorTurns, topiccontinuity.MaxPriorTurns, limits.MaxPriorTurns)
	}
	if limits.MaxTurnBytes < topiccontinuity.MinTurnBytes || limits.MaxTurnBytes > topiccontinuity.MaxTurnBytes {
		return fmt.Errorf("limits.max_turn_bytes must be in [%d, %d], got %d",
			topiccontinuity.MinTurnBytes, topiccontinuity.MaxTurnBytes, limits.MaxTurnBytes)
	}
	if limits.MaxInputBytes > topiccontinuity.MaxInputBytes && rule.inputBytesDerived() {
		return fmt.Errorf("limits.max_input_bytes derived as (max_prior_turns + 1) * max_turn_bytes = %d exceeds %d; "+
			"set an explicit limits.max_input_bytes", limits.MaxInputBytes, topiccontinuity.MaxInputBytes)
	}
	if limits.MaxInputBytes < topiccontinuity.MinInputBytes || limits.MaxInputBytes > topiccontinuity.MaxInputBytes {
		return fmt.Errorf("limits.max_input_bytes must be in [%d, %d], got %d",
			topiccontinuity.MinInputBytes, topiccontinuity.MaxInputBytes, limits.MaxInputBytes)
	}
	if limits.MaxInputBytes < limits.MaxTurnBytes {
		return fmt.Errorf("limits.max_input_bytes (%d) must be at least limits.max_turn_bytes (%d)",
			limits.MaxInputBytes, limits.MaxTurnBytes)
	}
	return nil
}
