package config

import (
	"fmt"
	"strings"

	lingua "github.com/pemistahl/lingua-go"
)

// validateLanguageContracts keeps the public language signal aligned with the
// classifier contract: names are lowercase ISO 639-1 codes supported by the
// detector, not arbitrary rule identifiers.
func validateLanguageContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	for index, rule := range cfg.LanguageRules {
		code := strings.TrimSpace(rule.Name)
		if code != strings.ToLower(code) || lingua.GetIsoCode639_1FromValue(code) == lingua.UnknownIsoCode639_1 {
			return fmt.Errorf(
				"routing.signals.language[%d].name %q must be a supported lowercase ISO 639-1 code such as en, zh, es, fr, ja, or de",
				index,
				rule.Name,
			)
		}
		if rule.Threshold < 0 || rule.Threshold > 1 {
			return fmt.Errorf("routing.signals.language[%d].threshold must be between 0 and 1", index)
		}
	}
	return nil
}
