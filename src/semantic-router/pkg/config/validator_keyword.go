package config

import (
	"fmt"
	"strings"
)

func validateKeywordContracts(cfg *RouterConfig) (err error) {
	defer func() { err = cfg.inRecipe(err) }()
	for _, rule := range cfg.KeywordRules {
		if err := ValidateKeywordRuleContract(rule); err != nil {
			return fmt.Errorf("signals.keywords[%q]: %w", rule.Name, err)
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
	if len(rule.Keywords) == 0 {
		// An AND rule over no keyword matches every text by accident, while
		// the catch-all is an explicit empty pattern: keywords: [""].
		return fmt.Errorf("rule needs at least one keyword; an empty pattern is the catch-all")
	}
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

// KeywordRuleFindsText reports why a rule cannot say where in a text it
// matched, which a rule must for masking to hide what it found: an OR rule of
// literal or regex keywords locates each occurrence, AND and NOR match on a
// combination or an absence, BM25, n-gram and fuzzy matching score
// similarity, and a count or density matches on how many there are.
func KeywordRuleFindsText(rule KeywordRule) error {
	switch method := strings.ToLower(rule.Method); {
	case rule.Operator != "OR":
		return fmt.Errorf("operator %s matches on which keywords occur, not where; use OR", rule.Operator)
	case method != "" && method != "regex":
		return fmt.Errorf("method %s scores similarity and does not locate a match; use literal keywords or method: regex", rule.Method)
	case rule.FuzzyMatch:
		return fmt.Errorf("fuzzy_match does not locate a match")
	case rule.Count != nil || rule.Density != nil:
		return fmt.Errorf("a count or density matches on how often keywords occur, not on one occurrence")
	}
	return nil
}
