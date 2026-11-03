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
		if rule.matchesToolResults() {
			// A single text is user content; tool-result rules never see it.
			continue
		}
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
	return c.MatchAllWithHistory(text, "")
}

// MatchAllWithHistory is MatchAll with an optional conversation history.
// Rules with includeHistory match against historyText (prior user messages
// plus the current message) when it is non-empty; every other rule matches
// against currentText only. Rules with source: tool_result never match here
// because no tool results are supplied.
func (c *KeywordClassifier) MatchAllWithHistory(currentText, historyText string) ([]KeywordRuleMatch, error) {
	return c.MatchAllWithSources(KeywordMatchInput{CurrentText: currentText, HistoryText: historyText})
}

// KeywordMatchInput is the request content MatchAllWithSources evaluates.
// Each rule reads exactly one of the three fields, selected by its
// configuration.
type KeywordMatchInput struct {
	// CurrentText is the current user message.
	CurrentText string
	// HistoryText is the prior user messages joined with the current one. It
	// is read by include_history rules and ignored when empty.
	HistoryText string
	// ToolResultTexts are the textual tool results of the request in request
	// order, one string per tool result. They are read by source: tool_result
	// rules, each text on its own, and never by any other rule.
	ToolResultTexts []string
}

// MatchAllWithSources evaluates every keyword rule against the content its
// source selects and returns all matches in declaration order.
func (c *KeywordClassifier) MatchAllWithSources(input KeywordMatchInput) ([]KeywordRuleMatch, error) {
	if c == nil {
		return nil, nil
	}
	analysis := lexical.NewText(input.CurrentText)
	// historyAnalysis is built lazily, only if a scored rule needs it.
	var historyAnalysis *lexical.Text
	matches := make([]KeywordRuleMatch, 0)

	for _, rule := range c.rules {
		var match ruleMatch
		var err error
		switch {
		case rule.matchesToolResults():
			match, err = c.matchRuleAcrossTexts(input.ToolResultTexts, rule)
		case rule.includeHistory && input.HistoryText != "":
			if historyAnalysis == nil {
				historyAnalysis = lexical.NewText(input.HistoryText)
			}
			match, err = c.matchRule(input.HistoryText, historyAnalysis, rule)
		default:
			match, err = c.matchRule(input.CurrentText, analysis, rule)
		}
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

// matchRuleAcrossTexts evaluates one regex rule against several independent
// texts, so a phrase or an AND rule cannot be assembled from two tool results.
// An AND or OR rule matches when any text matches and reports the keywords
// matched across all matching texts; a NOR rule matches only when every text
// is free of its keywords. No text means no match, so NOR cannot match a
// request that carries no tool results. Configuration validation restricts
// source: tool_result to regex rules, which never read a lexical analysis.
func (c *KeywordClassifier) matchRuleAcrossTexts(texts []string, rule keywordRule) (ruleMatch, error) {
	if len(texts) == 0 {
		return ruleMatch{}, nil
	}
	nor := rule.operator == "NOR"
	combined := ruleMatch{ruleName: rule.name}
	seen := make(map[string]struct{})
	for _, text := range texts {
		match, err := c.matchRule(text, nil, rule)
		if err != nil {
			return ruleMatch{}, err
		}
		if !match.matched {
			if nor {
				return ruleMatch{}, nil
			}
			continue
		}
		combined.matched = true
		combined.totalKeywords = match.totalKeywords
		for _, keyword := range match.keywords {
			if _, ok := seen[keyword]; ok {
				continue
			}
			seen[keyword] = struct{}{}
			combined.keywords = append(combined.keywords, keyword)
		}
	}
	if combined.matched {
		combined.matchCount = len(combined.keywords)
	}
	return combined, nil
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
