package classification

import (
	"fmt"
	"regexp"
	"strings"
	"unicode"
)

// matchesWithCount checks if the text matches the given keyword rule.
func (c *KeywordClassifier) matchesWithCount(text string, rule preppedKeywordRule) (bool, []string, int, error) {
	regexpsToUse := regexpsForRule(rule)
	lowerTextWords := lowerWordsForRule(rule, text)

	switch rule.Operator {
	case "AND":
		return matchAND(text, rule, regexpsToUse, lowerTextWords)
	case "OR":
		return matchOR(text, rule, regexpsToUse, lowerTextWords)
	case "NOR":
		return matchNOR(text, rule, regexpsToUse, lowerTextWords)
	default:
		return false, nil, 0, fmt.Errorf("unsupported keyword rule operator: %q", rule.Operator)
	}
}

func regexpsForRule(rule preppedKeywordRule) []*regexp.Regexp {
	if rule.CaseSensitive {
		return rule.CompiledRegexpsCS
	}
	return rule.CompiledRegexpsCI
}

func lowerWordsForRule(rule preppedKeywordRule, text string) []string {
	if !rule.FuzzyMatch {
		return nil
	}
	return extractLowerWords(text)
}

func validateRegexp(ruleName string, idx int, re *regexp.Regexp) error {
	if re != nil {
		return nil
	}
	return fmt.Errorf("nil regular expression found in rule %q at index %d. This indicates a failed compilation during initialization", ruleName, idx)
}

func matchAND(text string, rule preppedKeywordRule, regexpsToUse []*regexp.Regexp, lowerTextWords []string) (bool, []string, int, error) {
	matchedKeywords := make([]string, 0, len(rule.OriginalKeywords))
	for i, re := range regexpsToUse {
		if err := validateRegexp(rule.Name, i, re); err != nil {
			return false, nil, 0, err
		}
		if re.MatchString(text) {
			matchedKeywords = append(matchedKeywords, rule.OriginalKeywords[i])
			continue
		}
		if hasFuzzyMatch(rule, i, lowerTextWords) {
			matchedKeywords = append(matchedKeywords, rule.OriginalKeywords[i]+" (fuzzy)")
			continue
		}
		return false, nil, 0, nil
	}
	return true, matchedKeywords, len(matchedKeywords), nil
}

func matchOR(text string, rule preppedKeywordRule, regexpsToUse []*regexp.Regexp, lowerTextWords []string) (bool, []string, int, error) {
	matchedKeywords := make([]string, 0, len(rule.OriginalKeywords))
	matchedSet := make(map[string]bool)
	for i, re := range regexpsToUse {
		if err := validateRegexp(rule.Name, i, re); err != nil {
			return false, nil, 0, err
		}
		keyword := rule.OriginalKeywords[i]
		if re.MatchString(text) {
			addMatchedKeyword(keyword, matchedSet, &matchedKeywords)
			continue
		}
		if hasFuzzyMatch(rule, i, lowerTextWords) {
			addMatchedKeyword(keyword+" (fuzzy)", matchedSet, &matchedKeywords)
		}
	}
	if len(matchedKeywords) == 0 {
		return false, nil, 0, nil
	}
	return true, matchedKeywords, len(matchedKeywords), nil
}

func matchNOR(text string, rule preppedKeywordRule, regexpsToUse []*regexp.Regexp, lowerTextWords []string) (bool, []string, int, error) {
	for i, re := range regexpsToUse {
		if err := validateRegexp(rule.Name, i, re); err != nil {
			return false, nil, 0, err
		}
		if re.MatchString(text) || hasFuzzyMatch(rule, i, lowerTextWords) {
			return false, nil, 0, nil
		}
	}
	return true, nil, 0, nil
}

func hasFuzzyMatch(rule preppedKeywordRule, idx int, lowerTextWords []string) bool {
	if !rule.FuzzyMatch || idx >= len(rule.LowercaseKeywords) {
		return false
	}
	return fuzzyMatch(rule.LowercaseKeywords[idx], lowerTextWords, rule.FuzzyThreshold)
}

func addMatchedKeyword(keyword string, matchedSet map[string]bool, matchedKeywords *[]string) {
	if matchedSet[keyword] {
		return
	}
	matchedSet[keyword] = true
	*matchedKeywords = append(*matchedKeywords, keyword)
}

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
