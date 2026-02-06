package classification

import (
	"fmt"
	"regexp"
	"strings"
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

	// Fuzzy matching fields
	FuzzyMatch        bool     // Enable approximate matching with Levenshtein distance
	FuzzyThreshold    int      // Maximum edit distance for fuzzy matching (default: 2)
	LowercaseKeywords []string // Pre-computed lowercase for fuzzy matching
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
			FuzzyMatch:       rule.FuzzyMatch,
			FuzzyThreshold:   rule.FuzzyThreshold,
		}

		// Set default fuzzy threshold if enabled but not specified
		if preppedRule.FuzzyMatch && preppedRule.FuzzyThreshold == 0 {
			preppedRule.FuzzyThreshold = 2 // Default: allow 2 character edits
		}

		// Pre-compute lowercase keywords for fuzzy matching
		if rule.FuzzyMatch {
			preppedRule.LowercaseKeywords = make([]string, len(rule.Keywords))
			for j, keyword := range rule.Keywords {
				preppedRule.LowercaseKeywords[j] = strings.ToLower(keyword)
			}
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
// Returns category, confidence (0.0-1.0), and error.
// Confidence is calculated as: 0.5 + (matchCount / totalKeywords * 0.5)
func (c *KeywordClassifier) Classify(text string) (string, float64, error) {
	category, _, matchCount, totalKeywords, err := c.ClassifyWithKeywordsAndCount(text)
	if err != nil || category == "" {
		return category, 0.0, err
	}

	if totalKeywords == 0 {
		return category, 1.0, nil // Edge case: no keywords defined
	}

	ratio := float64(matchCount) / float64(totalKeywords)
	confidence := 0.5 + (ratio * 0.5)

	return category, confidence, nil
}

// ClassifyWithKeywords performs keyword-based classification and returns the matched keywords.
func (c *KeywordClassifier) ClassifyWithKeywords(text string) (string, []string, error) {
	category, keywords, _, _, err := c.ClassifyWithKeywordsAndCount(text)
	return category, keywords, err
}

// ClassifyWithKeywordsAndCount performs keyword-based classification and returns:
// - category: the matched rule name (or "" if no match)
// - matchedKeywords: slice of keywords that matched
// - matchCount: number of keywords that matched
// - totalKeywords: total number of keywords in the matched rule
// - error: any error that occurred
func (c *KeywordClassifier) ClassifyWithKeywordsAndCount(text string) (string, []string, int, int, error) {
	for _, rule := range c.rules {
		matched, keywords, matchCount, err := c.matchesWithCount(text, rule)
		if err != nil {
			return "", nil, 0, 0, err
		}
		if matched {
			totalKeywords := len(rule.OriginalKeywords)
			if len(keywords) > 0 {
				logging.Infof("Keyword-based classification matched rule %q with keywords: %v (%d/%d matched)",
					rule.Name, keywords, matchCount, totalKeywords)
			} else {
				logging.Infof("Keyword-based classification matched rule %q with a NOR rule.", rule.Name)
			}
			return rule.Name, keywords, matchCount, totalKeywords, nil
		}
	}
	return "", nil, 0, 0, nil
}

// matchesWithCount checks if the text matches the given keyword rule.
func (c *KeywordClassifier) matchesWithCount(text string, rule preppedKeywordRule) (bool, []string, int, error) {
	var matchedKeywords []string
	var regexpsToUse []*regexp.Regexp

	if rule.CaseSensitive {
		regexpsToUse = rule.CompiledRegexpsCS
	} else {
		regexpsToUse = rule.CompiledRegexpsCI
	}

	// Pre-extract and lowercase words for fuzzy matching (only if enabled)
	var lowerTextWords []string
	if rule.FuzzyMatch {
		lowerTextWords = extractLowerWords(text)
	}

	// Check for matches based on the operator
	switch rule.Operator {
	case "AND":
		for i, re := range regexpsToUse {
			if re == nil {
				return false, nil, 0, fmt.Errorf("nil regular expression found in rule %q at index %d. This indicates a failed compilation during initialization", rule.Name, i)
			}

			// Try exact regex match first
			if re.MatchString(text) {
				matchedKeywords = append(matchedKeywords, rule.OriginalKeywords[i])
				continue
			}

			// Try fuzzy match if enabled
			if rule.FuzzyMatch && i < len(rule.LowercaseKeywords) {
				if fuzzyMatch(rule.LowercaseKeywords[i], lowerTextWords, rule.FuzzyThreshold) {
					matchedKeywords = append(matchedKeywords, rule.OriginalKeywords[i]+" (fuzzy)")
					continue
				}
			}

			return false, nil, 0, nil // One keyword missing = no match for AND
		}
		return true, matchedKeywords, len(matchedKeywords), nil

	case "OR":
		// Collect ALL matching keywords for confidence calculation
		matchedSet := make(map[string]bool) // Avoid duplicates

		for i, re := range regexpsToUse {
			if re == nil {
				return false, nil, 0, fmt.Errorf("nil regular expression found in rule for category %q at index %d. This indicates a failed compilation during initialization", rule.Name, i)
			}

			keyword := rule.OriginalKeywords[i]

			// Try exact regex match first
			if re.MatchString(text) {
				if !matchedSet[keyword] {
					matchedSet[keyword] = true
					matchedKeywords = append(matchedKeywords, keyword)
				}
				continue
			}

			// Try fuzzy match if enabled
			if rule.FuzzyMatch && i < len(rule.LowercaseKeywords) {
				if fuzzyMatch(rule.LowercaseKeywords[i], lowerTextWords, rule.FuzzyThreshold) {
					fuzzyKeyword := keyword + " (fuzzy)"
					if !matchedSet[keyword] && !matchedSet[fuzzyKeyword] {
						matchedSet[fuzzyKeyword] = true
						matchedKeywords = append(matchedKeywords, fuzzyKeyword)
					}
				}
			}
		}

		if len(matchedKeywords) > 0 {
			return true, matchedKeywords, len(matchedKeywords), nil
		}
		return false, nil, 0, nil

	case "NOR":
		for i, re := range regexpsToUse {
			if re == nil {
				return false, nil, 0, fmt.Errorf("nil regular expression found in rule for category %q at index %d. This indicates a failed compilation during initialization", rule.Name, i)
			}
			if re.MatchString(text) {
				return false, nil, 0, nil // Forbidden keyword found
			}

			// Check fuzzy match if enabled
			if rule.FuzzyMatch && i < len(rule.LowercaseKeywords) {
				if fuzzyMatch(rule.LowercaseKeywords[i], lowerTextWords, rule.FuzzyThreshold) {
					return false, nil, 0, nil // Forbidden keyword found via fuzzy
				}
			}
		}
		return true, nil, 0, nil // None of the forbidden keywords found

	default:
		return false, nil, 0, fmt.Errorf("unsupported keyword rule operator: %q", rule.Operator)
	}
}

// ----------- Fuzzy Matching -----------

// levenshteinDistance calculates the edit distance between two strings.
// Uses Wagner-Fischer dynamic programming approach with O(m*n) time complexity.
func levenshteinDistance(s1, s2 string) int {
	s1 = strings.ToLower(s1)
	s2 = strings.ToLower(s2)

	r1 := []rune(s1)
	r2 := []rune(s2)
	len1 := len(r1)
	len2 := len(r2)

	if len1 == 0 {
		return len2
	}
	if len2 == 0 {
		return len1
	}

	// Optimize space to O(min(m,n))
	if len1 > len2 {
		r1, r2 = r2, r1
		len1, len2 = len2, len1
	}

	prev := make([]int, len1+1)
	curr := make([]int, len1+1)

	for i := 0; i <= len1; i++ {
		prev[i] = i
	}

	for j := 1; j <= len2; j++ {
		curr[0] = j
		for i := 1; i <= len1; i++ {
			cost := 0
			if r1[i-1] != r2[j-1] {
				cost = 1
			}
			curr[i] = min(prev[i]+1, min(curr[i-1]+1, prev[i-1]+cost))
		}
		prev, curr = curr, prev
	}

	return prev[len1]
}

// fuzzyMatch checks if any word in text fuzzy-matches the keyword within threshold.
func fuzzyMatch(lowerKeyword string, lowerTextWords []string, threshold int) bool {
	for _, textWord := range lowerTextWords {
		if levenshteinDistance(textWord, lowerKeyword) <= threshold {
			return true
		}
	}
	return false
}

// extractLowerWords splits text into lowercase words for fuzzy matching.
func extractLowerWords(text string) []string {
	var words []string
	var currentWord strings.Builder

	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			currentWord.WriteRune(r)
		} else if currentWord.Len() > 0 {
			words = append(words, strings.ToLower(currentWord.String()))
			currentWord.Reset()
		}
	}

	if currentWord.Len() > 0 {
		words = append(words, strings.ToLower(currentWord.String()))
	}

	return words
}
