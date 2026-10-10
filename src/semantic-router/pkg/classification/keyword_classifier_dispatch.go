package classification

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification/lexical"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type ruleMatch struct {
	matched       bool
	ruleName      string
	keywords      []string
	matchCount    int
	totalKeywords int
}

// KeywordRuleMatch describes one matching keyword rule. MatchAll returns these
// in configuration order so callers can compose rule signals without changing
// the legacy first-match classification APIs.
type KeywordRuleMatch struct {
	RuleName      string
	Keywords      []string
	MatchCount    int
	TotalKeywords int
}

// ClassifyWithKeywordsAndCount performs keyword-based classification and returns:
// - category: the matched rule name (or "" if no match)
// - matchedKeywords: slice of keywords that matched
// - matchCount: number of keywords that matched
// - totalKeywords: total number of keywords in the matched rule
// - error: any error that occurred
//
// Rules are evaluated in the order they were defined in the config (first-match semantics).
func (c *KeywordClassifier) ClassifyWithKeywordsAndCount(text string) (string, []string, int, int, error) {
	if c == nil {
		return "", nil, 0, 0, nil
	}
	analysis := lexical.NewText(text)
	for _, rule := range c.rules {
		match, err := c.matchRule(text, analysis, rule)
		if err != nil {
			return "", nil, 0, 0, err
		}
		if match.matched {
			logRuleMatch(rule.method, match.ruleName, match.keywords, match.matchCount, match.totalKeywords)
			return match.ruleName, match.keywords, match.matchCount, match.totalKeywords, nil
		}
	}
	return "", nil, 0, 0, nil
}

// MatchAll evaluates every keyword rule and returns all matches in declaration
// order. Decision evaluation must use this method because a request can satisfy
// multiple named signals that participate in AND, OR, and NOT expressions.
func (c *KeywordClassifier) MatchAll(text string) ([]KeywordRuleMatch, error) {
	if c == nil {
		return nil, nil
	}
	analysis := lexical.NewText(text)
	matches := make([]KeywordRuleMatch, 0)
	for _, rule := range c.rules {
		match, err := c.matchRule(text, analysis, rule)
		if err != nil {
			return nil, err
		}
		if !match.matched {
			continue
		}
		logRuleMatch(rule.method, match.ruleName, match.keywords, match.matchCount, match.totalKeywords)
		matches = append(matches, KeywordRuleMatch{
			RuleName:      match.ruleName,
			Keywords:      match.keywords,
			MatchCount:    match.matchCount,
			TotalKeywords: match.totalKeywords,
		})
	}
	return matches, nil
}

// matchRule evaluates one rule. analysis shares tokenization between the
// BM25 and n-gram rules of one call.
func (c *KeywordClassifier) matchRule(text string, analysis *lexical.Text, rule keywordRule) (ruleMatch, error) {
	if rule.scored != nil {
		match, ok := rule.scored.Match(analysis)
		if !ok {
			return ruleMatch{}, nil
		}
		return ruleMatch{
			matched:       true,
			ruleName:      rule.name,
			keywords:      match.Keywords,
			matchCount:    match.MatchCount,
			totalKeywords: match.TotalKeywords,
		}, nil
	}
	matched, keywords, matchCount, err := c.matchesWithCount(text, *rule.regex)
	if err != nil || !matched {
		return ruleMatch{}, err
	}
	return ruleMatch{
		matched:       true,
		ruleName:      rule.name,
		keywords:      keywords,
		matchCount:    matchCount,
		totalKeywords: len(rule.regex.OriginalKeywords),
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
