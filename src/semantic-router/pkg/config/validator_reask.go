package config

import "fmt"

func validateReaskContracts(cfg *RouterConfig) error {
	for _, rule := range cfg.ReaskRules {
		resolved := rule.WithDefaults()
		if resolved.Name == "" {
			return fmt.Errorf("routing.signals.reasks: name cannot be empty")
		}
		if resolved.Threshold < 0 || resolved.Threshold > 1 {
			return fmt.Errorf("routing.signals.reasks[%q]: threshold must be between 0 and 1", rule.Name)
		}
		if resolved.LookbackTurns < 1 {
			return fmt.Errorf("routing.signals.reasks[%q]: lookback_turns must be >= 1", rule.Name)
		}
	}
	return nil
}
