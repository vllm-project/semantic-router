package classification

import (
	"context"
	"sync"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (c *Classifier) evaluateKeywordSignal(results *SignalResults, mu *sync.Mutex, text string) {
	start := time.Now()
	matches, err := c.keywordClassifier.MatchAll(text)
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

type categoryProbabilityFallbackPolicy interface {
	fallbackToTop1OnProbabilityError() bool
}

func categoryProbabilityFallbackAllowed(inference CategoryInference) bool {
	policy, ok := inference.(categoryProbabilityFallbackPolicy)
	return !ok || policy.fallbackToTop1OnProbabilityError()
}

func (c *Classifier) evaluateDomainSignal(ctx context.Context, results *SignalResults, mu *sync.Mutex, text string) {
	start := time.Now()
	domainResult, err := c.categoryInference.ClassifyWithProbabilities(ctx, text)
	if err != nil && categoryProbabilityFallbackAllowed(c.categoryInference) {
		// Fall back to Classify() (top-1 only) when ClassifyWithProbabilities is unavailable.
		logging.Debugf("[Signal Computation] ClassifyWithProbabilities unavailable, falling back to Classify: %v", err)
		basicResult, basicErr := c.categoryInference.Classify(ctx, text)
		if basicErr != nil {
			err = basicErr
		} else {
			domainResult = candle_binding.ClassResultWithProbs{
				Class:      basicResult.Class,
				Confidence: basicResult.Confidence,
			}
			err = nil
		}
	}
	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()

	categoryName := ""
	if err == nil {
		if name, ok := c.CategoryMapping.GetCategoryFromIndex(domainResult.Class); ok {
			categoryName = c.translateMMLUToGeneric(name)
		}
	}

	c.recordSignalExtraction(config.SignalTypeDomain, categoryName, latencySeconds)

	// Record metrics
	results.Metrics.Domain.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	if categoryName != "" && err == nil {
		results.Metrics.Domain.Confidence = float64(domainResult.Confidence)
	}
	logging.Debugf("[Signal Computation] Domain signal evaluation completed in %v", elapsed)

	if err != nil {
		logging.Errorf("domain rule evaluation failed: %v", err)
	} else {
		matched := c.matchDomainCategories(domainResult, categoryName)
		mu.Lock()
		for _, cat := range matched {
			c.recordSignalMatch(config.SignalTypeDomain, cat.Category)
			results.MatchedDomainRules = append(results.MatchedDomainRules, cat.Category)
			results.SignalConfidences["domain:"+cat.Category] = float64(cat.Probability)
		}
		mu.Unlock()
	}
}

func (c *Classifier) evaluateFactCheckSignal(results *SignalResults, mu *sync.Mutex, text string) {
	start := time.Now()
	factCheckResult, err := c.ClassifyFactCheck(text)
	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()

	// Determine which signal to output based on classification result
	signalName := "no_fact_check_needed"
	if err == nil && factCheckResult != nil && factCheckResult.NeedsFactCheck {
		signalName = "needs_fact_check"
	}

	// Record signal extraction metrics
	c.recordSignalExtraction(config.SignalTypeFactCheck, signalName, latencySeconds)

	// Record metrics (use microseconds for better precision)
	results.Metrics.FactCheck.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	if signalName != "" && err == nil && factCheckResult != nil {
		results.Metrics.FactCheck.Confidence = float64(factCheckResult.Confidence)
	}

	logging.Debugf("[Signal Computation] Fact-check signal evaluation completed in %v", elapsed)
	if err != nil {
		logging.Errorf("fact-check rule evaluation failed: %v", err)
	} else if factCheckResult != nil {
		// Check if this signal is defined in fact_check_rules
		for _, rule := range c.Config.FactCheckRules {
			if rule.Name == signalName {
				// Record signal match
				c.recordSignalMatch(config.SignalTypeFactCheck, rule.Name)

				mu.Lock()
				results.MatchedFactCheckRules = append(results.MatchedFactCheckRules, rule.Name)
				mu.Unlock()
				break
			}
		}
	}
}

func (c *Classifier) evaluateUserFeedbackSignal(results *SignalResults, mu *sync.Mutex, text string, hasPriorAssistantReply bool) {
	if !shouldEvaluateUserFeedbackSignal(hasPriorAssistantReply) {
		logging.Debugf("[Signal Computation] User feedback signal skipped: no prior assistant reply")
		return
	}

	start := time.Now()
	feedbackResult, err := c.ClassifyFeedback(text)
	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()

	// Use the feedback type directly as the signal name
	signalName := ""
	if err == nil && feedbackResult != nil {
		signalName = feedbackResult.FeedbackType
	}

	// Record signal extraction metrics
	c.recordSignalExtraction(config.SignalTypeUserFeedback, signalName, latencySeconds)

	// Record metrics (use microseconds for better precision)
	results.Metrics.UserFeedback.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	if signalName != "" && err == nil && feedbackResult != nil {
		results.Metrics.UserFeedback.Confidence = float64(feedbackResult.Confidence)
	}

	logging.Debugf("[Signal Computation] User feedback signal evaluation completed in %v", elapsed)
	if err != nil {
		logging.Errorf("user feedback rule evaluation failed: %v", err)
	} else if feedbackResult != nil {
		// Check if this signal is defined in user_feedback_rules
		for _, rule := range c.Config.UserFeedbackRules {
			if rule.Name == signalName {
				// Record signal match
				c.recordSignalMatch(config.SignalTypeUserFeedback, rule.Name)

				mu.Lock()
				results.MatchedUserFeedbackRules = append(results.MatchedUserFeedbackRules, rule.Name)
				mu.Unlock()
				break
			}
		}
	}
}

func (c *Classifier) evaluateReaskSignal(results *SignalResults, mu *sync.Mutex, currentUserText string, priorUserMessages []string) {
	start := time.Now()
	matchedRules, err := c.reaskClassifier.Classify(currentUserText, priorUserMessages)
	elapsed := time.Since(start)

	results.Metrics.Reask.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0

	logging.Debugf("[Signal Computation] Reask signal evaluation completed in %v", elapsed)
	if err != nil {
		logging.Errorf("reask rule evaluation failed: %v", err)
		return
	}
	if len(matchedRules) == 0 {
		return
	}

	bestConfidence := 0.0
	mu.Lock()
	for _, match := range matchedRules {
		if match.MinSimilarity > bestConfidence {
			bestConfidence = match.MinSimilarity
		}
		c.recordSignalExtraction(config.SignalTypeReask, match.RuleName, elapsed.Seconds())
		c.recordSignalMatch(config.SignalTypeReask, match.RuleName)
		results.MatchedReaskRules = append(results.MatchedReaskRules, match.RuleName)
		results.SignalConfidences["reask:"+match.RuleName] = match.MinSimilarity
		results.SignalValues["reask:"+match.RuleName] = float64(match.MatchedTurns)
	}
	results.Metrics.Reask.Confidence = bestConfidence
	mu.Unlock()
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
