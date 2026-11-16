package classification

import (
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// preppedKeywordRule stores preprocessed keywords for efficient regex matching.
type preppedKeywordRule struct {
	Name              string // Name is also used as category
	Operator          string
	CaseSensitive     bool
	OriginalKeywords  []string         // For logging/returning original case
	CompiledRegexpsCS []*regexp.Regexp // Compiled regex for case-sensitive
	CompiledRegexpsCI []*regexp.Regexp // Compiled regex for case-insensitive
	LiteralBoundaries []bool           // Check neighboring runes for literal non-CJK words

	FuzzyMatch        bool     // Enable approximate matching with Levenshtein distance
	FuzzyThreshold    int      // Maximum edit distance for fuzzy matching (default: 2)
	LowercaseKeywords []string // Pre-computed lowercase for fuzzy matching
}

// prepRegexRule creates a preppedKeywordRule from a config rule.
func prepRegexRule(rule config.KeywordRule) (preppedKeywordRule, error) {
	preppedRule := preppedKeywordRule{
		Name:             rule.Name,
		Operator:         rule.Operator,
		CaseSensitive:    rule.CaseSensitive,
		OriginalKeywords: rule.Keywords,
		FuzzyMatch:       rule.FuzzyMatch,
		FuzzyThreshold:   rule.FuzzyThreshold,
	}

	if preppedRule.FuzzyMatch && preppedRule.FuzzyThreshold == 0 {
		preppedRule.FuzzyThreshold = 2
	}

	if rule.FuzzyMatch {
		preppedRule.LowercaseKeywords = make([]string, len(rule.Keywords))
		for j, keyword := range rule.Keywords {
			preppedRule.LowercaseKeywords[j] = strings.ToLower(keyword)
		}
	}

	preppedRule.CompiledRegexpsCS = make([]*regexp.Regexp, len(rule.Keywords))
	preppedRule.CompiledRegexpsCI = make([]*regexp.Regexp, len(rule.Keywords))
	useExplicitRegex := strings.EqualFold(rule.Method, "regex")
	preppedRule.LiteralBoundaries = make([]bool, len(rule.Keywords))

	for j, keyword := range rule.Keywords {
		patternCS, patternCI := regexPatterns(keyword, useExplicitRegex)
		preppedRule.LiteralBoundaries[j] = !useExplicitRegex && literalKeywordNeedsBoundary(keyword)

		var err error
		preppedRule.CompiledRegexpsCS[j], err = regexp.Compile(patternCS)
		if err != nil {
			logging.Errorf("Failed to compile case-sensitive regex for keyword %q: %v", keyword, err)
			return preppedKeywordRule{}, err
		}

		preppedRule.CompiledRegexpsCI[j], err = regexp.Compile(patternCI)
		if err != nil {
			logging.Errorf("Failed to compile case-insensitive regex for keyword %q: %v", keyword, err)
			return preppedKeywordRule{}, err
		}
	}

	return preppedRule, nil
}

func regexPatterns(keyword string, useExplicitRegex bool) (string, string) {
	if useExplicitRegex {
		return keyword, "(?i)" + keyword
	}

	pattern := regexp.QuoteMeta(keyword)
	return pattern, "(?i)" + pattern
}

// Keep the legacy substring contract for punctuation-only literals. Word-like
// literals need neighboring-rune boundaries, except for continuous CJK text.
func literalKeywordNeedsBoundary(keyword string) bool {
	return !containsCJK(keyword) && strings.ContainsFunc(keyword, func(r rune) bool {
		return unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_'
	})
}

// Literal boundaries belong to the neighboring runes, not to the keyword's
// first/last character: CJK substrings match, the regex engine's Unicode case
// folding applies, and an underscore joins a word. Explicit regex patterns
// remain untouched. Advance one rune after a rejected match so an
// overlapping or later valid occurrence is still considered.
func matchesKeywordPattern(text string, rule preppedKeywordRule, index int, pattern *regexp.Regexp) bool {
	if !rule.LiteralBoundaries[index] {
		return pattern.MatchString(text)
	}
	_, _, found := nextLiteralMatch(text, 0, pattern)
	return found
}

// keywordPlaces lists a keyword's matches in text as byte ranges, one per
// place: a pattern's non-overlapping matches, and a literal keyword's matches
// that are words of their own.
func keywordPlaces(text string, rule preppedKeywordRule, index int, pattern *regexp.Regexp) [][]int {
	if !rule.LiteralBoundaries[index] {
		return pattern.FindAllStringIndex(text, -1)
	}
	var places [][]int
	for offset := 0; ; {
		start, end, found := nextLiteralMatch(text, offset, pattern)
		if !found {
			return places
		}
		places = append(places, []int{start, end})
		offset = end
	}
}

// nextLiteralMatch finds the first match of a literal keyword at or after
// offset that is a word of its own, and returns where it starts and ends.
func nextLiteralMatch(text string, offset int, pattern *regexp.Regexp) (int, int, bool) {
	for offset < len(text) {
		match := pattern.FindStringIndex(text[offset:])
		if match == nil {
			return 0, 0, false
		}
		start, end := offset+match[0], offset+match[1]
		if keywordBoundaryMatch(text, start, end) &&
			(start == 0 || text[start-1] != '_') && (end == len(text) || text[end] != '_') {
			return start, end, true
		}
		_, size := utf8.DecodeRuneInString(text[start:])
		offset = start + size
	}
	return 0, 0, false
}

// occurrences counts the places the rule's keywords occur in text and lists
// the keywords that occur.
func (r preppedKeywordRule) occurrences(text string) (int, []string) {
	count := 0
	var found []string
	for index, pattern := range regexpsForRule(r) {
		if occurrences := len(keywordPlaces(text, r, index, pattern)); occurrences > 0 {
			count += occurrences
			found = append(found, r.OriginalKeywords[index])
		}
	}
	return count, found
}
