package config

import (
	"fmt"
	"strings"
)

func validateKeywordContracts(cfg *RouterConfig) error {
	for _, rule := range cfg.KeywordRules {
		if err := ValidateKeywordRuleContract(rule); err != nil {
			return fmt.Errorf("routing.signals.keywords[%q]: %w", rule.Name, err)
		}
	}
	return nil
}

// ValidateKeywordRuleContract checks a keyword rule's count or density
// against its operator and method; the keyword classifier checks the other
// fields when it compiles the rule. The DSL compiler shares it.
//
// A count is the number of places the rule's keywords occur, which only
// literal and regex matching locate: BM25 and n-gram rules score how similar
// the text is to each keyword, and fuzzy matching compares words by edit
// distance.
func ValidateKeywordRuleContract(rule KeywordRule) error {
	measure, predicate := "count", rule.Count
	switch {
	case rule.Count != nil && rule.Density != nil:
		return fmt.Errorf("set count or density, not both")
	case rule.Density != nil:
		measure, predicate = "density", rule.Density
	case rule.Count == nil:
		return nil
	}
	if err := validateNumericPredicateContract(predicate); err != nil {
		return fmt.Errorf("%s: %w", measure, err)
	}
	if rule.Operator != "OR" {
		return fmt.Errorf("%s counts the keywords an OR rule finds; operator %q matches on which keywords occur", measure, rule.Operator)
	}
	if method := strings.ToLower(rule.Method); method != "" && method != "regex" {
		return fmt.Errorf("%s counts occurrences, which method %s does not locate; use literal keywords or method: regex", measure, rule.Method)
	}
	if rule.FuzzyMatch {
		return fmt.Errorf("%s counts occurrences, which fuzzy_match does not locate", measure)
	}
	return nil
}
