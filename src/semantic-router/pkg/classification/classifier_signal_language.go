package classification

import (
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func (c *Classifier) evaluateLanguageSignal(results *SignalResults, mu *sync.Mutex, text string) {
	start := time.Now()
	// Use the lowest configured threshold across all rules so a single
	// classification pass covers all rules. Rules with higher thresholds are
	// still checked at match time; rules with lower thresholds (or none) use
	// the built-in default. This preserves backward compatibility: when no rule
	// sets a threshold the behaviour is identical to the previous Classify() call.
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
	metrics.RecordSignalExtraction(config.SignalTypeLanguage, languageCode, latencySeconds)

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
			// If the rule has a custom threshold, enforce it here.
			if rule.Threshold > 0 && float32(languageResult.Confidence) < rule.Threshold {
				logging.Debugf("[Signal Computation] Language rule %q skipped: confidence %.2f < threshold %.2f",
					rule.Name, languageResult.Confidence, rule.Threshold)
				break
			}
			// Record signal match
			metrics.RecordSignalMatch(config.SignalTypeLanguage, rule.Name)

			mu.Lock()
			results.MatchedLanguageRules = append(results.MatchedLanguageRules, rule.Name)
			mu.Unlock()
			break
		}
	}
}

// lowestLanguageThreshold returns the smallest non-zero Threshold across all
// configured LanguageRules, or 0 if none is set. This value is passed to
// ClassifyWithThreshold so that a single lingua-go call covers all rules.
func lowestLanguageThreshold(rules []config.LanguageRule) float32 {
	var min float32
	for _, r := range rules {
		if r.Threshold > 0 && (min == 0 || r.Threshold < min) {
			min = r.Threshold
		}
	}
	return min
}
