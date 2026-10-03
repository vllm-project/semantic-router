package config

import (
	"fmt"
	"slices"
	"strings"
)

func validateActionContracts(cfg *RouterConfig) error {
	for index, rule := range cfg.ActionRules {
		if !slices.Contains(SupportedActions(), rule.Name) {
			return fmt.Errorf(
				"routing.signals.actions[%d].name %q must be one of %s",
				index,
				rule.Name,
				strings.Join(SupportedActions(), ", "),
			)
		}
	}
	return nil
}
