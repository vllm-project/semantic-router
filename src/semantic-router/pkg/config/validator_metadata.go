package config

import (
	"fmt"
	"strings"
)

func validateMetadataContracts(cfg *RouterConfig) error {
	seen := make(map[string]struct{}, len(cfg.MetadataRules))
	for i, rule := range cfg.MetadataRules {
		trimmedName := strings.TrimSpace(rule.Name)
		if trimmedName == "" {
			return fmt.Errorf("routing.signals.metadata[%d]: name is required", i)
		}
		if trimmedName != rule.Name {
			return fmt.Errorf(
				"routing.signals.metadata[%d]: name must not contain surrounding whitespace",
				i,
			)
		}
		normalizedName := strings.ToLower(rule.Name)
		if _, exists := seen[normalizedName]; exists {
			return fmt.Errorf("routing.signals.metadata[%d]: duplicate name %q", i, rule.Name)
		}
		seen[normalizedName] = struct{}{}
		trimmedKey := strings.TrimSpace(rule.Key)
		if trimmedKey == "" {
			return fmt.Errorf("routing.signals.metadata[%q]: key is required", rule.Name)
		}
		if trimmedKey != rule.Key {
			return fmt.Errorf(
				"routing.signals.metadata[%q]: key must not contain surrounding whitespace",
				rule.Name,
			)
		}
		comparators := 0
		if rule.Predicate.Equals != nil {
			comparators++
		}
		if len(rule.Predicate.In) > 0 {
			comparators++
		}
		if rule.Predicate.Exists != nil {
			comparators++
		}
		if comparators != 1 {
			return fmt.Errorf(
				"routing.signals.metadata[%q]: predicate must set exactly one of equals, in, or exists",
				rule.Name,
			)
		}
	}
	return nil
}
