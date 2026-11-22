package config

import "fmt"

// validateDomainContracts checks the model scores of the domains rules; the
// rules themselves are application rules (validateApplicationSignalContracts).
func validateDomainContracts(cfg *RouterConfig) error {
	for _, category := range cfg.Categories {
		if err := validateCategoryModelScores(category); err != nil {
			return err
		}
	}
	return nil
}

func formatDomainSuggestion(value string) string {
	suggestion := SuggestSupportedRoutingDomainName(value)
	if suggestion == "" || suggestion == value {
		return ""
	}
	return fmt.Sprintf("; did you mean %q?", suggestion)
}
