package classification

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// KeywordClassifier implements keyword-based classification logic.
type KeywordClassifier struct {
	rules []config.KeywordRule
}

// NewKeywordClassifier creates a new KeywordClassifier.
func NewKeywordClassifier(rules []config.KeywordRule) *KeywordClassifier {
	return &KeywordClassifier{rules: rules}
}

// Classify performs keyword-based classification on the given text.
func (c *KeywordClassifier) Classify(text string) (string, float64, error) {
	for _, rule := range c.rules {
		if c.matches(text, rule) {
			return rule.Category, 1.0, nil
		}
	}
	return "", 0.0, nil
}

// matches checks if the text matches the given keyword rule.
func (c *KeywordClassifier) matches(text string, rule config.KeywordRule) bool {
	// Default to case-insensitive matching if not specified
	caseSensitive := rule.CaseSensitive

	// Prepare text for matching
	preparedText := text
	if !caseSensitive {
		preparedText = strings.ToLower(text)
	}

	// Check for matches based on the operator
	switch rule.Operator {
	case "AND":
		for _, keyword := range rule.Keywords {
			preparedKeyword := keyword
			if !caseSensitive {
				preparedKeyword = strings.ToLower(keyword)
			}
			if !strings.Contains(preparedText, preparedKeyword) {
				return false
			}
		}
		return true
	case "OR":
		for _, keyword := range rule.Keywords {
			preparedKeyword := keyword
			if !caseSensitive {
				preparedKeyword = strings.ToLower(keyword)
			}
			if strings.Contains(preparedText, preparedKeyword) {
				return true
			}
		}
		return false
	case "NOR":
		for _, keyword := range rule.Keywords {
			preparedKeyword := keyword
			if !caseSensitive {
				preparedKeyword = strings.ToLower(keyword)
			}
			if strings.Contains(preparedText, preparedKeyword) {
				return false
			}
		}
		return true
	default:
		return false
	}
}