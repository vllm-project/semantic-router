package config

import (
	"errors"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const legacyLatencyMixedConfigErr = "deprecated latency signal routing config cannot be used with decision.algorithm.type=latency_aware; remove either legacy latency config (signals.latency_rules / conditions.type=latency) or latency_aware decisions"

// normalizeLegacyLatencyRouting migrates old latency signal-based routing config
// to decision.algorithm.type=latency_aware when migration is guaranteed to be lossless.
//
// legacy latency compatibility is now temporary and will be removed after the backward-compatibility period.
//
// Rules:
//   - mixed new+old latency config is rejected
//   - old-only config is auto-migrated only for strict AND + single latency condition cases
//   - non-lossless legacy patterns return explicit errors
func normalizeLegacyLatencyRouting(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}

	hasLegacy := hasLegacyLatencyRoutingConfig(cfg)
	hasLatencyAware := hasAnyLatencyAwareDecision(cfg.Decisions)
	if hasLegacy && hasLatencyAware {
		return errors.New(legacyLatencyMixedConfigErr)
	}

	migratedCount := 0
	for i := range cfg.Decisions {
		decision := &cfg.Decisions[i]

		latencyConditionIdx := -1
		latencyConditionCount := 0
		for idx, condition := range decision.Rules.Conditions {
			if strings.EqualFold(strings.TrimSpace(condition.Type), SignalTypeLatency) {
				latencyConditionCount++
				if latencyConditionIdx == -1 {
					latencyConditionIdx = idx
				}
			}
		}

		if latencyConditionCount == 0 {
			continue
		}

		legacyAlgorithmOnError := ""
		if decision.Algorithm != nil {
			normalizedAlgoType := strings.ToLower(strings.TrimSpace(decision.Algorithm.Type))
			if normalizedAlgoType != "static" {
				algoType := strings.TrimSpace(decision.Algorithm.Type)
				if algoType == "" {
					algoType = "<empty>"
				}
				return fmt.Errorf("decision '%s': legacy latency condition conflicts with decision.algorithm.type=%s; only static can be auto-migrated to latency_aware", decision.Name, algoType)
			}
			legacyAlgorithmOnError = decision.Algorithm.OnError
		}

		if latencyConditionCount > 1 {
			return fmt.Errorf("decision '%s': multiple legacy latency conditions are not supported for auto-migration", decision.Name)
		}

		if strings.TrimSpace(decision.Rules.Operator) != "AND" {
			return fmt.Errorf("decision '%s': legacy latency condition with rules.operator=%q cannot be auto-migrated; only AND is supported", decision.Name, decision.Rules.Operator)
		}

		latencyCondition := decision.Rules.Conditions[latencyConditionIdx]
		latencyRule, err := resolveLegacyLatencyRule(cfg.Signals.LatencyRules, latencyCondition.Name)
		if err != nil {
			return fmt.Errorf("decision '%s': %w", decision.Name, err)
		}

		remainingConditions := removeConditionAt(decision.Rules.Conditions, latencyConditionIdx)
		if len(remainingConditions) == 0 {
			return fmt.Errorf("decision '%s': legacy latency condition cannot be auto-migrated because no non-latency conditions remain", decision.Name)
		}

		decision.Rules.Conditions = remainingConditions
		decision.Algorithm = &AlgorithmConfig{
			Type:    "latency_aware",
			OnError: legacyAlgorithmOnError,
			LatencyAware: &LatencyAwareAlgorithmConfig{
				TPOTPercentile: latencyRule.TPOTPercentile,
				TTFTPercentile: latencyRule.TTFTPercentile,
				Description:    latencyRule.Description,
			},
		}

		migratedCount++
		logging.Warnf("DEPRECATED: decision '%s' uses conditions.type=latency (name=%s), which is deprecated and will be removed in a future release. Auto-migrated to decision.algorithm.type=latency_aware.", decision.Name, latencyCondition.Name)
	}

	if migratedCount > 0 {
		cfg.Signals.LatencyRules = nil
		logging.Warnf("DEPRECATED: signals.latency_rules is deprecated and will be removed in a future release. Auto-migrated to decision.algorithm.type=latency_aware.")
	}

	return nil
}

func resolveLegacyLatencyRule(rules []LatencyRule, name string) (*LatencyRule, error) {
	foundIdx := -1
	for i := range rules {
		if rules[i].Name == name {
			if foundIdx != -1 {
				return nil, fmt.Errorf("legacy latency rule name '%s' is duplicated in signals.latency_rules", name)
			}
			foundIdx = i
		}
	}

	if foundIdx == -1 {
		return nil, fmt.Errorf("legacy latency condition references unknown latency rule '%s'", name)
	}

	return &rules[foundIdx], nil
}

func removeConditionAt(conditions []RuleCondition, idx int) []RuleCondition {
	result := make([]RuleCondition, 0, len(conditions)-1)
	result = append(result, conditions[:idx]...)
	result = append(result, conditions[idx+1:]...)
	return result
}
