package classification

import (
	"fmt"
	"regexp"
	"unicode"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// preppedKeywordRule stores preprocessed keywords for efficient matching.
type preppedKeywordRule struct {
	Name              string // Name is also used as category
	Operator          string
	CaseSensitive     bool
	OriginalKeywords  []string         // For logging/returning original case
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
			return nil, fmt.Errorf("unsupported keyword rule operator: %q for rule %q", rule.Operator, rule.Name)
		}

		preppedRule := preppedKeywordRule{
			Name:             rule.Name,
			Operator:         rule.Operator,
			CaseSensitive:    rule.CaseSensitive,
			OriginalKeywords: rule.Keywords,
		}

		// Compile regexps for both case-sensitive and case-insensitive
		preppedRule.CompiledRegexpsCS = make([]*regexp.Regexp, len(rule.Keywords))
		preppedRule.CompiledRegexpsCI = make([]*regexp.Regexp, len(rule.Keywords))

		for j, keyword := range rule.Keywords {
			quotedKeyword := regexp.QuoteMeta(keyword)
			// Conditionally add word boundaries. If the keyword contains at least one word character,
			// apply word boundaries. However, skip word boundaries for Chinese characters since \b
			// doesn't work with non-ASCII characters.
			hasWordChar := false
			hasChinese := false
			for _, r := range keyword {
				if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' {
					hasWordChar = true
				}
				// Check if the character is Chinese (CJK Unified Ideographs)
				if unicode.Is(unicode.Han, r) {
					hasChinese = true
				}
				if hasWordChar && hasChinese {
					break
				}
			}

			patternCS := quotedKeyword
			patternCI := "(?i)" + quotedKeyword

			// Only add word boundaries for non-Chinese keywords
			if hasWordChar && !hasChinese {
				patternCS = "\\b" + patternCS + "\\b"
				patternCI = "(?i)\\b" + quotedKeyword + "\\b"
			}

			var err error
			preppedRule.CompiledRegexpsCS[j], err = regexp.Compile(patternCS)
			if err != nil {
				logging.Errorf("Failed to compile case-sensitive regex for keyword %q: %v", keyword, err)
				return nil, err
			}

			preppedRule.CompiledRegexpsCI[j], err = regexp.Compile(patternCI)
			if err != nil {
				logging.Errorf("Failed to compile case-insensitive regex for keyword %q: %v", keyword, err)
				return nil, err
			}
		}
		preppedRules[i] = preppedRule
	}
	return &KeywordClassifier{rules: preppedRules}, nil
}

// Classify performs keyword-based classification on the given text.
func (c *KeywordClassifier) Classify(text string) (string, float64, error) {
	category, _, err := c.ClassifyWithKeywords(text)
	return category, 1.0, err
}

// ClassifyWithKeywords performs keyword-based classification and returns the matched keywords.
func (c *KeywordClassifier) ClassifyWithKeywords(text string) (string, []string, error) {
	for _, rule := range c.rules {
		matched, keywords, err := c.matches(text, rule) // Error handled
		if err != nil {
			return "", nil, err // Propagate error
		}
		if matched {
			if len(keywords) > 0 {
				logging.Infof("Keyword-based classification matched rule %q with keywords: %v", rule.Name, keywords)
			} else {
				logging.Infof("Keyword-based classification matched rule %q with a NOR rule.", rule.Name)
			}
			return rule.Name, keywords, nil
		}
	}
	return "", nil, nil
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
				return false, nil, fmt.Errorf("nil regular expression found in rule %q at index %d. This indicates a failed compilation during initialization", rule.Name, i)
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
				return false, nil, fmt.Errorf("nil regular expression found in rule for category %q at index %d. This indicates a failed compilation during initialization", rule.Name, i)
			}
			if re.MatchString(text) {
				return true, []string{rule.OriginalKeywords[i]}, nil
			}
		}
		return false, nil, nil
	case "NOR":
		for i, re := range regexpsToUse {
			if re == nil {
				return false, nil, fmt.Errorf("nil regular expression found in rule for category %q at index %d. This indicates a failed compilation during initialization", rule.Name, i)
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
