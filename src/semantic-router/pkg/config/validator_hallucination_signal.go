package config

import (
	"fmt"
	"strings"
)

// validateHallucinationSignalContracts checks the hallucination rules of one
// routing profile.
func validateHallucinationSignalContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	seen := make(map[string]struct{}, len(cfg.HallucinationRules))
	for _, rule := range cfg.HallucinationRules {
		name := strings.TrimSpace(rule.Name)
		if name == "" {
			return fmt.Errorf("routing.signals.hallucination: every rule needs a name")
		}
		if _, dup := seen[name]; dup {
			return fmt.Errorf("routing.signals.hallucination: duplicate rule name %q", name)
		}
		seen[name] = struct{}{}
	}
	return nil
}
