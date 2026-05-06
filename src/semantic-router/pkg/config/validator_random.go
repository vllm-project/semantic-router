package config

import "fmt"

func validateRandomContracts(cfg *RouterConfig) error {
	seen := make(map[string]struct{}, len(cfg.RandomRules))
	for _, rule := range cfg.RandomRules {
		if rule.Name == "" {
			return fmt.Errorf("routing.signals.random: name cannot be empty")
		}
		if _, ok := seen[rule.Name]; ok {
			return fmt.Errorf("routing.signals.random[%q]: duplicate rule name", rule.Name)
		}
		seen[rule.Name] = struct{}{}
	}
	return nil
}
