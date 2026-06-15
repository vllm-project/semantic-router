package classification

import (
	"regexp"
	"strings"
	"unicode"

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

	quotedKeyword := regexp.QuoteMeta(keyword)
	hasWordChar := false
	hasChinese := false
	for _, r := range keyword {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' {
			hasWordChar = true
		}
		if unicode.Is(unicode.Han, r) {
			hasChinese = true
		}
		if hasWordChar && hasChinese {
			break
		}
	}

	patternCS := quotedKeyword
	patternCI := "(?i)" + quotedKeyword
	if hasWordChar && !hasChinese {
		patternCS = "\\b" + patternCS + "\\b"
		patternCI = "(?i)\\b" + quotedKeyword + "\\b"
	}
	return patternCS, patternCI
}
