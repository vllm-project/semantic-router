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

	for j, keyword := range rule.Keywords {
		patternCS, patternCI := regexPatterns(keyword, useExplicitRegex)

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
	// Han keywords retain substring matching because written Chinese does not
	// delimit words with spaces. Other literal keywords use Unicode word
	// boundaries only at ends that are themselves letters, digits or '_'.
	for _, r := range keyword {
		if unicode.Is(unicode.Han, r) {
			return pattern, "(?i)" + pattern
		}
	}
	first, _ := utf8.DecodeRuneInString(keyword)
	last, _ := utf8.DecodeLastRuneInString(keyword)
	isWord := func(r rune) bool { return unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' }
	if isWord(first) {
		pattern = `(?:^|[^\p{L}\p{N}_])` + pattern
	}
	if isWord(last) {
		pattern += `(?:$|[^\p{L}\p{N}_])`
	}
	return pattern, "(?i)" + pattern
}
