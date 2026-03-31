package classification

import (
	"fmt"
	"regexp"
	"strings"
	"unicode"

	nlp_binding "github.com/vllm-project/semantic-router/nlp-binding"
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

	// Fuzzy matching fields
	FuzzyMatch        bool     // Enable approximate matching with Levenshtein distance
	FuzzyThreshold    int      // Maximum edit distance for fuzzy matching (default: 2)
	LowercaseKeywords []string // Pre-computed lowercase for fuzzy matching
}

// KeywordClassifier implements keyword-based classification logic.
// It supports three matching methods per rule: regex (default), bm25, and ngram.
// BM25 and N-gram rules are dispatched to Rust-backed classifiers via nlp-binding.
type KeywordClassifier struct {
	regexRules []preppedKeywordRule // Regex-based rules (original behavior)

	// Rust-backed classifiers via nlp-binding FFI
	bm25Classifier  *nlp_binding.BM25Classifier
	ngramClassifier *nlp_binding.NgramClassifier

	// Track which rules use which method for ordered evaluation
	ruleOrder []ruleRef
}

// ruleRef tracks the method and index for ordered rule evaluation.
type ruleRef struct {
	method string // "regex", "bm25", "ngram"
	name   string // rule name for logging
}

type ruleMatch struct {
	matched       bool
	ruleName      string
	keywords      []string
	matchCount    int
	totalKeywords int
}

// NewKeywordClassifier creates a new KeywordClassifier.
// Rules with method "bm25" or "ngram" are dispatched to Rust-backed classifiers;
// all others (including default/empty method) use the original regex engine.
func NewKeywordClassifier(cfgRules []config.KeywordRule) (*KeywordClassifier, error) {
	kc := &KeywordClassifier{}

	var hasBM25, hasNgram bool

	for _, rule := range cfgRules {
		// Validate operator
		switch rule.Operator {
		case "AND", "OR", "NOR":
			// Valid
		default:
			return nil, fmt.Errorf("unsupported keyword rule operator: %q for rule %q", rule.Operator, rule.Name)
		}

		method := strings.ToLower(rule.Method)
		if method == "" {
			method = "regex"
		}

		switch method {
		case "bm25":
			hasBM25 = true
		case "ngram":
			hasNgram = true
		case "regex":
			// default
		default:
			return nil, fmt.Errorf("unsupported keyword rule method: %q for rule %q (valid: regex, bm25, ngram)", rule.Method, rule.Name)
		}
	}

	// Initialize Rust-backed classifiers only if needed
	if hasBM25 {
		kc.bm25Classifier = nlp_binding.NewBM25Classifier()
	}
	if hasNgram {
		kc.ngramClassifier = nlp_binding.NewNgramClassifier()
	}

	// Process each rule according to its method
	for _, rule := range cfgRules {
		method := strings.ToLower(rule.Method)
		if method == "" {
			method = "regex"
		}

		switch method {
		case "bm25":
			threshold := rule.BM25Threshold
			if threshold == 0 {
				threshold = 0.1 // default BM25 threshold
			}
			err := kc.bm25Classifier.AddRule(
				rule.Name, rule.Operator, rule.Keywords, threshold, rule.CaseSensitive,
			)
			if err != nil {
				return nil, fmt.Errorf("failed to add BM25 rule %q: %w", rule.Name, err)
			}
			kc.ruleOrder = append(kc.ruleOrder, ruleRef{method: "bm25", name: rule.Name})
			logging.ComponentDebugEvent("classifier", "keyword_rule_registered", map[string]interface{}{
				"rule":      rule.Name,
				"method":    "bm25",
				"threshold": threshold,
				"keywords":  len(rule.Keywords),
			})

		case "ngram":
			threshold := rule.NgramThreshold
			if threshold == 0 {
				threshold = 0.4 // default n-gram similarity threshold
			}
			arity := rule.NgramArity
			if arity == 0 {
				arity = 3 // default trigram
			}
			err := kc.ngramClassifier.AddRule(
				rule.Name, rule.Operator, rule.Keywords, threshold, rule.CaseSensitive, arity,
			)
			if err != nil {
				return nil, fmt.Errorf("failed to add N-gram rule %q: %w", rule.Name, err)
			}
			kc.ruleOrder = append(kc.ruleOrder, ruleRef{method: "ngram", name: rule.Name})
			logging.ComponentDebugEvent("classifier", "keyword_rule_registered", map[string]interface{}{
				"rule":      rule.Name,
				"method":    "ngram",
				"threshold": threshold,
				"arity":     arity,
				"keywords":  len(rule.Keywords),
			})

		case "regex":
			preppedRule, err := prepRegexRule(rule)
			if err != nil {
				return nil, err
			}
			kc.regexRules = append(kc.regexRules, preppedRule)
			kc.ruleOrder = append(kc.ruleOrder, ruleRef{method: "regex", name: rule.Name})
			logging.ComponentDebugEvent("classifier", "keyword_rule_registered", map[string]interface{}{
				"rule":     rule.Name,
				"method":   "regex",
				"keywords": len(rule.Keywords),
				"fuzzy":    rule.FuzzyMatch,
			})
		}
	}

	return kc, nil
}

// Free releases Rust-side resources. Call when the classifier is no longer needed.
func (c *KeywordClassifier) Free() {
	if c.bm25Classifier != nil {
		c.bm25Classifier.Free()
	}
	if c.ngramClassifier != nil {
		c.ngramClassifier.Free()
	}
}

// prepRegexRule creates a preppedKeywordRule from a config rule (original regex logic).
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

// Classify performs keyword-based classification on the given text.
// Returns category, confidence (0.0-1.0), and error.
// For regex: confidence = 0.5 + (matchCount / totalKeywords * 0.5)
// For BM25/N-gram: confidence derived from match scores
func (c *KeywordClassifier) Classify(text string) (string, float64, error) {
	category, _, matchCount, totalKeywords, err := c.ClassifyWithKeywordsAndCount(text)
	if err != nil || category == "" {
		return category, 0.0, err
	}

	if totalKeywords == 0 {
		return category, 1.0, nil
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
//
// Rules are evaluated in the order they were defined in the config (first-match semantics),
// regardless of method. Each rule is dispatched to its respective engine.
func (c *KeywordClassifier) ClassifyWithKeywordsAndCount(text string) (string, []string, int, int, error) {
	regexIdx := 0

	for _, ref := range c.ruleOrder {
		match, err := c.classifyRule(text, ref, &regexIdx)
		if err != nil {
			return "", nil, 0, 0, err
		}
		if match.matched {
			logRuleMatch(ref.method, match.ruleName, match.keywords, match.matchCount, match.totalKeywords)
			return match.ruleName, match.keywords, match.matchCount, match.totalKeywords, nil
		}
	}
	return "", nil, 0, 0, nil
}

func (c *KeywordClassifier) classifyRule(text string, ref ruleRef, regexIdx *int) (ruleMatch, error) {
	switch ref.method {
	case "bm25":
		result := c.bm25Classifier.Classify(text)
		if !result.Matched || result.RuleName != ref.name {
			return ruleMatch{}, nil
		}
		return ruleMatch{
			matched:       true,
			ruleName:      result.RuleName,
			keywords:      result.MatchedKeywords,
			matchCount:    result.MatchCount,
			totalKeywords: result.TotalKeywords,
		}, nil
	case "ngram":
		result := c.ngramClassifier.Classify(text)
		if !result.Matched || result.RuleName != ref.name {
			return ruleMatch{}, nil
		}
		return ruleMatch{
			matched:       true,
			ruleName:      result.RuleName,
			keywords:      result.MatchedKeywords,
			matchCount:    result.MatchCount,
			totalKeywords: result.TotalKeywords,
		}, nil
	case "regex":
		return c.classifyRegexRule(text, regexIdx)
	default:
		return ruleMatch{}, nil
	}
}

func (c *KeywordClassifier) classifyRegexRule(text string, regexIdx *int) (ruleMatch, error) {
	if *regexIdx >= len(c.regexRules) {
		return ruleMatch{}, nil
	}
	rule := c.regexRules[*regexIdx]
	*regexIdx++
	matched, keywords, matchCount, err := c.matchesWithCount(text, rule)
	if err != nil {
		return ruleMatch{}, err
	}
	if !matched {
		return ruleMatch{}, nil
	}
	return ruleMatch{
		matched:       true,
		ruleName:      rule.Name,
		keywords:      keywords,
		matchCount:    matchCount,
		totalKeywords: len(rule.OriginalKeywords),
	}, nil
}

func logRuleMatch(method, ruleName string, keywords []string, matchCount, totalKeywords int) {
	prefix := "Keyword-based"
	switch method {
	case "bm25":
		prefix = "BM25 keyword"
	case "ngram":
		prefix = "N-gram keyword"
	}
	if len(keywords) > 0 {
		logging.Infof("%s classification matched rule %q with keywords: %v (%d/%d matched)",
			prefix, ruleName, keywords, matchCount, totalKeywords)
		return
	}
	logging.Infof("%s classification matched rule %q with a NOR rule.", prefix, ruleName)
}

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
