package classification

import (
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification/lexical"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type ruleMatch struct {
	matched       bool
	ruleName      string
	keywords      []string
	matchCount    int
	totalKeywords int
	// value is a counting rule's count or density; measured marks one.
	value    float64
	measured bool
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
			logRuleMatch(rule.method, match)
			return match.ruleName, match.keywords, match.matchCount, match.totalKeywords, nil
		}
	}
	return "", nil, 0, 0, nil
}

// MatchAll evaluates every keyword rule and returns all matches in declaration
// order. Decision evaluation must use this method because a request can
// satisfy multiple named signals that participate in AND, OR, and NOT
// expressions.
func (c *KeywordClassifier) MatchAll(text string) ([]KeywordRuleMatch, error) {
	matches, _, err := c.MatchAllWithHistory(text, "")
	return matches, err
}

// MatchAllWithHistory is MatchAll with an optional conversation history.
// Rules with includeHistory match against historyText (prior user messages
// plus the current message) when it is non-empty; every other rule matches
// against currentText only. It also returns, by rule name, the count or
// density each counting rule measured, whether or not the rule matched.
func (c *KeywordClassifier) MatchAllWithHistory(currentText, historyText string) ([]KeywordRuleMatch, map[string]float64, error) {
	if c == nil {
		return nil, nil, nil
	}
	analysis := lexical.NewText(currentText)
	// historyAnalysis is built lazily, only if some rule opts into history.
	var historyAnalysis *lexical.Text
	matches := make([]KeywordRuleMatch, 0)
	var values map[string]float64

	for _, rule := range c.rules {
		text, ruleAnalysis := currentText, analysis
		if rule.includeHistory && historyText != "" {
			if historyAnalysis == nil {
				historyAnalysis = lexical.NewText(historyText)
			}
			text, ruleAnalysis = historyText, historyAnalysis
		}
		match, err := c.matchRule(text, ruleAnalysis, rule)
		if err != nil {
			return nil, nil, err
		}
		if match.measured {
			if values == nil {
				values = make(map[string]float64)
			}
			values[rule.name] = match.value
		}
		if !match.matched {
			continue
		}
		logRuleMatch(rule.method, match)
		matches = append(matches, KeywordRuleMatch{
			RuleName:      match.ruleName,
			Keywords:      match.keywords,
			MatchCount:    match.matchCount,
			TotalKeywords: match.totalKeywords,
		})
	}
	return matches, values, nil
}

// matchRule evaluates one rule. analysis shares tokenization between the
// BM25 and n-gram rules of one call.
func (c *KeywordClassifier) matchRule(text string, analysis *lexical.Text, rule keywordRule) (ruleMatch, error) {
	if rule.measure != nil {
		return rule.measureMatch(text), nil
	}
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

// measureMatch counts the rule's keyword occurrences in text and matches
// when the count, or the density per 1,000 characters, satisfies the rule's
// predicate; an empty text has a density of zero.
func (r keywordRule) measureMatch(text string) ruleMatch {
	count, keywords := r.regex.occurrences(text)
	value := float64(count)
	if r.measure.density {
		value = 0
		if characters := utf8.RuneCountInString(text); characters > 0 {
			value = 1000 * float64(count) / float64(characters)
		}
	}
	return ruleMatch{
		matched:       predicateMatches(value, r.measure.predicate),
		ruleName:      r.name,
		keywords:      keywords,
		matchCount:    len(keywords),
		totalKeywords: len(r.regex.OriginalKeywords),
		value:         value,
		measured:      true,
	}
}

func logRuleMatch(method string, match ruleMatch) {
	prefix := "Keyword-based"
	switch method {
	case "bm25":
		prefix = "BM25 keyword"
	case "ngram":
		prefix = "N-gram keyword"
	}
	switch {
	case match.measured:
		logging.Infof("%s classification matched rule %q on its count or density %g, with keywords: %v",
			prefix, match.ruleName, match.value, match.keywords)
	case len(match.keywords) > 0:
		logging.Infof("%s classification matched rule %q with keywords: %v (%d/%d matched)",
			prefix, match.ruleName, match.keywords, match.matchCount, match.totalKeywords)
	default:
		logging.Infof("%s classification matched rule %q with a NOR rule.", prefix, match.ruleName)
	}
}
