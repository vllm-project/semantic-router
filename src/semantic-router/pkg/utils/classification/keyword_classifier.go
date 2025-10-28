package classification

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
)

// preppedKeywordRule stores preprocessed keywords for efficient matching.
type preppedKeywordRule struct {
	Category         string
	Operator         string
	CaseSensitive    bool
	OriginalKeywords []string // For logging/returning original case
	LowerKeywords    []string // For case-insensitive matching
}

// KeywordClassifier implements keyword-based classification logic.
type KeywordClassifier struct {
	rules []preppedKeywordRule // Store preprocessed rules
}

// NewKeywordClassifier creates a new KeywordClassifier.
func NewKeywordClassifier(cfgRules []config.KeywordRule) *KeywordClassifier {
	preppedRules := make([]preppedKeywordRule, len(cfgRules))
	for i, rule := range cfgRules {
		preppedRule := preppedKeywordRule{
			Category:         rule.Category,
			Operator:         rule.Operator,
			CaseSensitive:    rule.CaseSensitive,
			OriginalKeywords: rule.Keywords,
		}
		if !rule.CaseSensitive {
			preppedRule.LowerKeywords = make([]string, len(rule.Keywords))
			for j, keyword := range rule.Keywords {
				preppedRule.LowerKeywords[j] = strings.ToLower(keyword)
			}
		}
		preppedRules[i] = preppedRule
	}
	return &KeywordClassifier{rules: preppedRules}
}

// Classify performs keyword-based classification on the given text.
func (c *KeywordClassifier) Classify(text string) (string, float64, error) {
	for _, rule := range c.rules {
		if matched, keywords := c.matches(text, rule); matched {
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
func (c *KeywordClassifier) matches(text string, rule preppedKeywordRule) (bool, []string) {
	var matchedKeywords []string

	// Prepare text for matching
	preparedText := text
	if !rule.CaseSensitive {
		preparedText = strings.ToLower(text)
	}

	// Determine which set of keywords to use
	keywordsToMatch := rule.OriginalKeywords
	if !rule.CaseSensitive {
		keywordsToMatch = rule.LowerKeywords
	}

	// Check for matches based on the operator
	switch rule.Operator {
	case "AND":
		for i, keyword := range keywordsToMatch {
			if !strings.Contains(preparedText, keyword) {
				return false, nil
			}
			matchedKeywords = append(matchedKeywords, rule.OriginalKeywords[i]) // Append original keyword for logging
		}
		return true, matchedKeywords
	case "OR":
		for i, keyword := range keywordsToMatch {
			if strings.Contains(preparedText, keyword) {
				return true, []string{rule.OriginalKeywords[i]} // Append original keyword for logging
			}
		}
		return false, nil
	case "NOR":
		for _, keyword := range keywordsToMatch {
			if strings.Contains(preparedText, keyword) {
				return false, nil
			}
		}
		return true, matchedKeywords // Return true with an empty slice
	default:
		observability.Warnf("Unsupported keyword rule operator: %q", rule.Operator)
		return false, nil
	}
}
