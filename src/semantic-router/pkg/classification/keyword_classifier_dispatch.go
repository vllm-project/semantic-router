package classification

import (
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification/lexical"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
		if !rule.readsRequest {
			// A single text is the user's turn, which this rule leaves out.
			continue
		}
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
// density each counting rule measured, whether or not the rule matched. No
// tool results are supplied, so a rule that reads only them never matches.
func (c *KeywordClassifier) MatchAllWithHistory(currentText, historyText string) ([]KeywordRuleMatch, map[string]float64, error) {
	return c.MatchAllWithSources(KeywordMatchInput{CurrentText: currentText, HistoryText: historyText})
}

// KeywordMatchInput is the request content MatchAllWithSources evaluates.
// Each rule reads the parts its configuration names.
type KeywordMatchInput struct {
	// CurrentText is the current user message.
	CurrentText string
	// HistoryText is the prior user messages joined with the current one. It
	// is read by include_history rules and ignored when empty.
	HistoryText string
	// PriorText is the prior user messages without the current one. It is
	// read by include_history rules that leave the current turn out
	// (input.request: false) and ignored when empty.
	PriorText string
	// ToolResultTexts are the textual tool results of the request in request
	// order, one string per tool result. They are read by input.tool_results
	// rules, each text on its own, and by no other rule.
	ToolResultTexts []string
	// ToolResultsIncomplete is whether the request carries a tool result the
	// Router could not read as text, which an input.tool_results rule
	// resolves as its on_unscanned says.
	ToolResultsIncomplete bool
}

// userText is the user text a rule reads, and whether there is any: the
// current turn, the earlier turns with it (include_history), or the earlier
// turns alone (input.request: false).
func (input KeywordMatchInput) userText(rule keywordRule) (string, bool) {
	switch {
	case rule.readsRequest && rule.includeHistory && input.HistoryText != "":
		return input.HistoryText, true
	case rule.readsRequest:
		return input.CurrentText, true
	case rule.includeHistory:
		return input.PriorText, input.PriorText != ""
	}
	return "", false
}

// MatchAllWithSources evaluates every keyword rule against the content its
// input selects and returns all matches in declaration order, with the count
// or density each counting rule measured.
func (c *KeywordClassifier) MatchAllWithSources(input KeywordMatchInput) ([]KeywordRuleMatch, map[string]float64, error) {
	if c == nil {
		return nil, nil, nil
	}
	// The scored rules share one tokenization per text they read.
	analyses := make(map[string]*lexical.Text, 3)
	analysisOf := func(text string) *lexical.Text {
		if analysis, ok := analyses[text]; ok {
			return analysis
		}
		analysis := lexical.NewText(text)
		analyses[text] = analysis
		return analysis
	}
	matches := make([]KeywordRuleMatch, 0)
	var values map[string]float64

	for _, rule := range c.rules {
		text, hasText := input.userText(rule)
		var match ruleMatch
		var err error
		switch {
		case rule.toolResults:
			texts := input.ToolResultTexts
			if hasText {
				texts = append([]string{text}, texts...)
			}
			var found bool
			match, found, err = c.matchRuleAcrossTexts(texts, rule)
			if err == nil && input.ToolResultsIncomplete {
				match = rule.applyOnUnscanned(match, found)
			}
		case !hasText:
			// The rule leaves the current turn out and no earlier turn exists.
			continue
		default:
			match, err = c.matchRule(text, analysisOf(text), rule)
		}
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

// matchRuleAcrossTexts evaluates one literal or regex rule against several
// independent texts, so a phrase or an AND rule cannot be assembled from two
// tool results, and reports whether any text held one of the rule's
// keywords. An AND or OR rule matches when any text matches and reports the
// keywords matched across all matching texts; a NOR rule matches only when
// every text is free of its keywords. No text means no match, so NOR cannot
// match a request that carries no tool results. Configuration validation
// keeps the scored methods, which read a lexical analysis, off the tool
// results.
func (c *KeywordClassifier) matchRuleAcrossTexts(texts []string, rule keywordRule) (match ruleMatch, found bool, err error) {
	if len(texts) == 0 {
		return ruleMatch{}, false, nil
	}
	nor := rule.operator == "NOR"
	combined := ruleMatch{ruleName: rule.name}
	seen := make(map[string]struct{})
	for _, text := range texts {
		match, err := c.matchRule(text, nil, rule)
		if err != nil {
			return ruleMatch{}, false, err
		}
		if nor {
			if !match.matched {
				// A keyword occurred, which a NOR rule reports as no match.
				return ruleMatch{}, true, nil
			}
			combined.matched, combined.totalKeywords = true, match.totalKeywords
			continue
		}
		if !match.matched {
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
	return combined, !nor && combined.matched, nil
}

// applyOnUnscanned is the rule's match when the Router could not read one of
// the request's tool results as text. A keyword found in the text read
// counts whatever on_unscanned says; when none was found, best_effort (the
// default) keeps the match decided on the text read, match matches the rule,
// and no_match does not.
func (r keywordRule) applyOnUnscanned(match ruleMatch, found bool) ruleMatch {
	if found {
		return match
	}
	switch r.onUnscanned {
	case config.KeywordUnscannedMatch:
		if !match.matched {
			logging.Infof("Keyword rule %q matched on a tool result the Router could not read as text (on_unscanned: match)", r.name)
			match = ruleMatch{matched: true, ruleName: r.name, totalKeywords: match.totalKeywords}
		}
	case config.KeywordUnscannedNoMatch:
		match = ruleMatch{}
	}
	return match
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
		logging.Infof("%s classification matched rule %q without a keyword: a NOR rule, or an unread tool result.", prefix, match.ruleName)
	}
}
