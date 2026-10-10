package config

import (
	"fmt"
	"strings"
)

func validateAgenticFactsSignalContracts(cfg *RouterConfig) error {
	seen := make(map[string]struct{}, len(cfg.AgenticFactsRules))
	for i, rule := range cfg.AgenticFactsRules {
		trimmedName := strings.TrimSpace(rule.Name)
		if trimmedName == "" {
			return fmt.Errorf("routing.signals.agentic_facts[%d]: name is required", i)
		}
		if trimmedName != rule.Name {
			return fmt.Errorf(
				"routing.signals.agentic_facts[%d]: name must not contain surrounding whitespace",
				i,
			)
		}
		normalizedName := strings.ToLower(rule.Name)
		if _, exists := seen[normalizedName]; exists {
			return fmt.Errorf("routing.signals.agentic_facts[%d]: duplicate name %q", i, rule.Name)
		}
		seen[normalizedName] = struct{}{}

		switch rule.Field {
		case AgenticFactsFieldDelegatedRole, AgenticFactsFieldTaskPhase:
		default:
			return fmt.Errorf(
				"routing.signals.agentic_facts[%q]: field must be %q or %q, got %q",
				rule.Name, AgenticFactsFieldDelegatedRole, AgenticFactsFieldTaskPhase, rule.Field,
			)
		}

		comparators := 0
		if rule.Predicate.Equals != nil {
			comparators++
		}
		if len(rule.Predicate.In) > 0 {
			comparators++
		}
		if comparators != 1 {
			return fmt.Errorf(
				"routing.signals.agentic_facts[%q]: predicate must set exactly one of equals or in",
				rule.Name,
			)
		}
	}
	return nil
}
