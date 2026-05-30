package classification

import (
	nlp_binding "github.com/vllm-project/semantic-router/nlp-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type ruleMatch struct {
	matched       bool
	ruleName      string
	keywords      []string
	matchCount    int
	totalKeywords int
}

// classifyState holds per-call lazy caches so the BM25 / N-gram classifiers
// are invoked at most once per ClassifyWithKeywordsAndCount call, regardless
// of how many BM25 / N-gram rule refs appear in ruleOrder.
//
// Without this cache, an outer loop with N BM25 refs invokes the full BM25
// classify N times, and each call internally iterates all rules until first
// match, yielding O(N^2) work plus N CString allocations of the full prompt.
type classifyState struct {
	regexIdx    int
	bm25Cached  bool
	bm25Result  nlp_binding.MatchResult
	ngramCached bool
	ngramResult nlp_binding.MatchResult
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
	if c == nil {
		return "", nil, 0, 0, nil
	}
	state := classifyState{}

	for _, ref := range c.ruleOrder {
		match, err := c.classifyRule(text, ref, &state)
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

func (c *KeywordClassifier) classifyRule(text string, ref ruleRef, state *classifyState) (ruleMatch, error) {
	switch ref.method {
	case "bm25":
		if !state.bm25Cached {
			state.bm25Result = c.bm25Classifier.Classify(text)
			state.bm25Cached = true
		}
		result := state.bm25Result
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
		if !state.ngramCached {
			state.ngramResult = c.ngramClassifier.Classify(text)
			state.ngramCached = true
		}
		result := state.ngramResult
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
		return c.classifyRegexRule(text, &state.regexIdx)
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
