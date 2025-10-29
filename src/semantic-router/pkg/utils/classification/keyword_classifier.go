package classification

import (
	"fmt"
	"regexp"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
)

// preppedKeywordRule stores preprocessed keywords for efficient matching.
type preppedKeywordRule struct {
	Category          string
	Operator          string
	CaseSensitive     bool
	OriginalKeywords  []string // For logging/returning original case
	CompiledRegexpsCS []*regexp.Regexp // Compiled regex for case-sensitive
	CompiledRegexpsCI []*regexp.Regexp // Compiled regex for case-insensitive
}

// KeywordClassifier implements keyword-based classification logic.
type KeywordClassifier struct {
	rules []preppedKeywordRule // Store preprocessed rules
}

// NewKeywordClassifier creates a new KeywordClassifier.
func NewKeywordClassifier(cfgRules []config.KeywordRule) (*KeywordClassifier, error) {
	preppedRules := make([]preppedKeywordRule, len(cfgRules))
	for i, rule := range cfgRules {
		// Validate operator
		switch rule.Operator {
		case "AND", "OR", "NOR":
			// Valid operator
		default:
			return nil, fmt.Errorf("unsupported keyword rule operator: %q for category %q", rule.Operator, rule.Category)
		}

		preppedRule := preppedKeywordRule{
			Category:        rule.Category,
			Operator:        rule.Operator,
			CaseSensitive:   rule.CaseSensitive,
			OriginalKeywords: rule.Keywords,
		}

		// Compile regexps for both case-sensitive and case-insensitive
		preppedRule.CompiledRegexpsCS = make([]*regexp.Regexp, len(rule.Keywords))
		preppedRule.CompiledRegexpsCI = make([]*regexp.Regexp, len(rule.Keywords))

		for j, keyword := range rule.Keywords {
			// Escape any regex special characters in the keyword and add word boundaries
			patternCS := "\\b" + regexp.QuoteMeta(keyword) + "\\b"
			patternCI := "(?i)\\b" + regexp.QuoteMeta(keyword) + "\\b" // (?i) for case-insensitive

			var err error
			preppedRule.CompiledRegexpsCS[j], err = regexp.Compile(patternCS)
			if err != nil {
				observability.Errorf("Failed to compile case-sensitive regex for keyword %q: %v", keyword, err)
				return nil, err
			}

			preppedRule.CompiledRegexpsCI[j], err = regexp.Compile(patternCI)
			if err != nil {
				observability.Errorf("Failed to compile case-insensitive regex for keyword %q: %v", keyword, err)
				return nil, err
			}
		}
		preppedRules[i] = preppedRule
	}
	return &KeywordClassifier{rules: preppedRules}, nil
}

// Classify performs keyword-based classification on the given text.
func (c *KeywordClassifier) Classify(text string) (string, float64, error) {
	for _, rule := range c.rules {
		matched, keywords, err := c.matches(text, rule) // Error handled
		if err != nil {
			return "", 0.0, err // Propagate error
		}
		if matched {
			if len(keywords) > 0 {
				observability.Infof("Keyword-based classification matched category %q with keywords: %v", rule.Category, keywords)
			} else {
				observability.Infof("Keyword-based classification matched category %q with a NOR rule.", rule.Category)
			}
			return rule.Category, 1.0, nil
		}
	}
	return "", 0.0, nil
}

// matches checks if the text matches the given keyword rule.
func (c *KeywordClassifier) matches(text string, rule preppedKeywordRule) (bool, []string, error) {
	var matchedKeywords []string
	var regexpsToUse []*regexp.Regexp

	if rule.CaseSensitive {
		regexpsToUse = rule.CompiledRegexpsCS
	} else {
		regexpsToUse = rule.CompiledRegexpsCI
	}

	// Check for matches based on the operator
	switch rule.Operator {
	case "AND":
		for i, re := range regexpsToUse {
			if re == nil {
				return false, nil, fmt.Errorf("nil regular expression found in rule for category %q at index %d. This indicates a failed compilation during initialization", rule.Category, i)
			}
			if !re.MatchString(text) {
				return false, nil, nil
			}
			matchedKeywords = append(matchedKeywords, rule.OriginalKeywords[i])
		}
		return true, matchedKeywords, nil
	case "OR":
		for i, re := range regexpsToUse {
			if re == nil {
				return false, nil, fmt.Errorf("nil regular expression found in rule for category %q at index %d. This indicates a failed compilation during initialization", rule.Category, i)
			}
			if re.MatchString(text) {
				return true, []string{rule.OriginalKeywords[i]}, nil
			}
		}
		return false, nil, nil
	case "NOR":
		for i, re := range regexpsToUse {
			if re == nil {
				return false, nil, fmt.Errorf("nil regular expression found in rule for category %q at index %d. This indicates a failed compilation during initialization", rule.Category, i)
			}
			if re.MatchString(text) {
				return false, nil, nil
			}
		}
		return true, matchedKeywords, nil
	default:
		return false, nil, fmt.Errorf("unsupported keyword rule operator: %q", rule.Operator)
	}
}


