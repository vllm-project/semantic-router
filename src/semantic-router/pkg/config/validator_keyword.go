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

// KeywordUnscanned* are the on_unscanned policies of a keyword rule that
// reads the tool results, named as the pii rule's are.
const (
	KeywordUnscannedBestEffort = SystemOneUnscannedBestEffort
	KeywordUnscannedMatch      = SystemOneUnscannedMatch
	KeywordUnscannedNoMatch    = SystemOneUnscannedNoMatch
)

// ValidateKeywordRuleContract checks a keyword rule's input, on_unscanned and
// count or density against its operator and method; the keyword classifier
// checks the other fields when it compiles the rule. The DSL compiler shares
// it.
//
// A count is the number of places the rule's keywords occur, which only
// literal and regex matching locate: BM25 and n-gram rules score how similar
// the text is to each keyword, and fuzzy matching compares words by edit
// distance. A tool result is matched on its own with literal or regex
// keywords, which recognize the shape of a secret; the scored methods rank
// user prose, and a count has one text to count in.
func ValidateKeywordRuleContract(rule KeywordRule) error {
	if err := validateKeywordInputContract(rule); err != nil {
		return err
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

func validateKeywordInputContract(rule KeywordRule) error {
	if !rule.Input.ReadsRequest() && !rule.Input.ToolResults && !rule.IncludeHistory {
		return fmt.Errorf("input.request: false leaves the current user turn out, so the rule reads input.tool_results or include_history")
	}
	if rule.OnUnscanned != "" && !rule.Input.ToolResults {
		return fmt.Errorf("on_unscanned decides a tool result the Router cannot read as text, so the rule reads input.tool_results")
	}
	switch rule.OnUnscanned {
	case "", KeywordUnscannedBestEffort, KeywordUnscannedMatch, KeywordUnscannedNoMatch:
	default:
		return fmt.Errorf("on_unscanned %q is not best_effort, match or no_match", rule.OnUnscanned)
	}
	if !rule.Input.ToolResults {
		return nil
	}
	if method := strings.ToLower(rule.Method); method != "" && method != "regex" {
		return fmt.Errorf("input.tool_results matches each tool result with literal or regex keywords; method %s scores user prose", rule.Method)
	}
	if rule.FuzzyMatch {
		return fmt.Errorf("input.tool_results matches each tool result with literal or regex keywords, which fuzzy_match does not")
	}
	if rule.Count != nil || rule.Density != nil {
		return fmt.Errorf("input.tool_results matches on which keywords occur in a tool result; count and density count one text")
	}
	return nil
}
