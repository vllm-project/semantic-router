package classification

import (
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// evaluateKeywordSignal evaluates keyword rules against the current text only.
func (c *Classifier) evaluateKeywordSignal(results *SignalResults, mu *sync.Mutex, text string) {
	c.evaluateKeywordSignalWithSources(results, mu, KeywordMatchInput{CurrentText: text}, nil)
}

// evaluateKeywordSignalWithSources evaluates keyword rules against the content
// each rule's configuration names: the current user message, the prior user
// messages for include_history rules, and the request's textual tool results
// for input.tool_results rules, with whether one of them went unread. The
// history texts are built here from priorUserMessages, only if a rule reads
// them. A counting rule publishes its count or density as its value, matched
// or not.
func (c *Classifier) evaluateKeywordSignalWithSources(results *SignalResults, mu *sync.Mutex, input KeywordMatchInput, priorUserMessages []string) {
	start := time.Now()
	if len(priorUserMessages) > 0 && c.keywordClassifier.HasHistoryRules() {
		input.HistoryText = keywordHistoryText(input.CurrentText, priorUserMessages)
		input.PriorText = keywordHistoryText("", priorUserMessages)
	}
	matches, values, err := c.keywordClassifier.MatchAllWithSources(input)
	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()

	primaryCategory := ""
	if len(matches) > 0 {
		primaryCategory = matches[0].RuleName
	}
	// Extraction latency is measured once per classifier invocation. Match
	// counters below still retain the individual rule labels.
	c.recordSignalExtraction(config.SignalTypeKeyword, primaryCategory, latencySeconds)

	// Record metrics (use microseconds for better precision)
	results.Metrics.Keyword.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	results.Metrics.Keyword.Confidence = 1.0 // Rule-based, always 1.0

	logging.Debugf("[Signal Computation] Keyword signal evaluation completed in %v", elapsed)
	if err != nil {
		logging.Errorf("keyword rule evaluation failed: %v", err)
		return
	}

	for _, match := range matches {
		c.recordSignalMatch(config.SignalTypeKeyword, match.RuleName)
	}

	mu.Lock()
	defer mu.Unlock()
	for name, value := range values {
		results.SignalValues[signalConfidenceKey(config.SignalTypeKeyword, name)] = value
	}
	seenKeywords := make(map[string]struct{}, len(results.MatchedKeywords))
	for _, keyword := range results.MatchedKeywords {
		seenKeywords[keyword] = struct{}{}
	}
	for _, match := range matches {
		results.MatchedKeywordRules = append(results.MatchedKeywordRules, match.RuleName)
		for _, keyword := range match.Keywords {
			if _, seen := seenKeywords[keyword]; seen {
				continue
			}
			seenKeywords[keyword] = struct{}{}
			results.MatchedKeywords = append(results.MatchedKeywords, keyword)
		}
	}
}

func (c *Classifier) evaluateContextSignal(
	results *SignalResults,
	mu *sync.Mutex,
	contextText string,
	contextTokenFloor int,
) {
	start := time.Now()
	matchedRules, count, err := c.contextClassifier.ClassifyWithTokenFloor(
		contextText,
		contextTokenFloor,
	)
	elapsed := time.Since(start)

	// Record metrics (use microseconds for better precision)
	results.Metrics.Context.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	results.Metrics.Context.Confidence = 1.0 // Rule-based, always 1.0

	logging.Debugf("[Signal Computation] Context signal evaluation completed in %v (count=%d)", elapsed, count)
	if err != nil {
		logging.Errorf("context rule evaluation failed: %v", err)
	} else {
		mu.Lock()
		results.MatchedContextRules = matchedRules
		results.TokenCount = count
		mu.Unlock()
	}
}
