package config

import (
	"fmt"
	"slices"
)

const PIISourceToolResult = "tool_result"

var validPIISources = []string{"", PIISourceToolResult}

// validatePIIContracts rejects source values that the runtime does not
// understand. An omitted source preserves the legacy user/history behavior.
func validatePIIContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	for _, rule := range cfg.PIIRules {
		if !slices.Contains(validPIISources, rule.Source) {
			return fmt.Errorf(
				"routing.signals.pii %q: unknown source %q; valid values are %q or omitted",
				rule.Name,
				rule.Source,
				PIISourceToolResult,
			)
		}
	}
	return nil
}
