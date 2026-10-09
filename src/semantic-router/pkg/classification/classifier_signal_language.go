package classification

import (
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (c *Classifier) evaluateLanguageSignal(results *SignalResults, mu *sync.Mutex, text string) {
	start := time.Now()
	// Use the lowest effective threshold so a single classification pass covers
	// all rules. Each rule's effective threshold is still checked at match time.
	threshold := lowestLanguageThreshold(c.Config.LanguageRules)
	languageResult, err := c.languageClassifier.ClassifyWithThreshold(text, threshold)
	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()

	// Use the language code directly as the signal name
	languageCode := ""
	if err == nil && languageResult != nil {
		languageCode = languageResult.LanguageCode
	}

	// Record signal extraction metrics
	c.recordSignalExtraction(config.SignalTypeLanguage, languageCode, latencySeconds)

	// Record metrics (use microseconds for better precision)
	results.Metrics.Language.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	if languageCode != "" && err == nil && languageResult != nil {
		results.Metrics.Language.Confidence = languageResult.Confidence
	}

	logging.Debugf("[Signal Computation] Language signal evaluation completed in %v", elapsed)
	if err != nil {
		logging.Errorf("language rule evaluation failed: %v", err)
	} else if languageResult != nil {
		// Check if this language code is defined in language_rules and
		// whether the detected confidence meets the per-rule threshold.
		for _, rule := range c.Config.LanguageRules {
			if rule.Name != languageCode {
				continue
			}
			threshold := effectiveLanguageThreshold(rule.Threshold)
			if float32(languageResult.Confidence) < threshold {
				logging.Debugf("[Signal Computation] Language rule %q skipped: confidence %.2f < threshold %.2f",
					rule.Name, languageResult.Confidence, threshold)
				break
			}
			// Record signal match
			c.recordSignalMatch(config.SignalTypeLanguage, rule.Name)

			mu.Lock()
			results.MatchedLanguageRules = append(results.MatchedLanguageRules, rule.Name)
			mu.Unlock()
			break
		}
	}
}

func effectiveLanguageThreshold(threshold float32) float32 {
	if threshold <= 0 {
		return defaultLanguageThreshold
	}
	return threshold
}

// lowestLanguageThreshold returns the smallest effective threshold across all
// configured LanguageRules, or 0 if there are no rules. This value is passed to
// ClassifyWithThreshold so that a single lingua-go call covers all rules.
func lowestLanguageThreshold(rules []config.LanguageRule) float32 {
	var min float32
	for _, r := range rules {
		threshold := effectiveLanguageThreshold(r.Threshold)
		if min == 0 || threshold < min {
			min = threshold
		}
	}
	return min
}
