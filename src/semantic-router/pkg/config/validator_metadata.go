package config

import (
	"fmt"
	"strings"
)

func validateMetadataContracts(cfg *RouterConfig) error {
	seen := make(map[string]struct{}, len(cfg.MetadataRules))
	for i, rule := range cfg.MetadataRules {
		if strings.TrimSpace(rule.Name) == "" {
			return fmt.Errorf("routing.signals.metadata[%d]: name is required", i)
		}
		if _, exists := seen[rule.Name]; exists {
			return fmt.Errorf("routing.signals.metadata[%d]: duplicate name %q", i, rule.Name)
		}
		seen[rule.Name] = struct{}{}
		if strings.TrimSpace(rule.Key) == "" {
			return fmt.Errorf("routing.signals.metadata[%q]: key is required", rule.Name)
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
