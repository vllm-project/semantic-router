package classification

import (
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func (c *Classifier) evaluatePreferenceSignal(results *SignalResults, mu *sync.Mutex, text string) {
	start := time.Now()
	contentBytes, _ := json.Marshal(text)
	conversationJSON := fmt.Sprintf(`[{"role":"user","content":%s}]`, contentBytes)

	preferenceResult, err := c.preferenceClassifier.Classify(conversationJSON)
	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()

	// Use the preference name directly as the signal name
	preferenceName := ""
	if err == nil && preferenceResult != nil {
		preferenceName = preferenceResult.Preference
	}

	// Record signal extraction metrics
	metrics.RecordSignalExtraction(config.SignalTypePreference, preferenceName, latencySeconds)

	// Record metrics (use microseconds for better precision)
	results.Metrics.Preference.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	if preferenceName != "" && err == nil && preferenceResult != nil && preferenceResult.Confidence > 0 {
		results.Metrics.Preference.Confidence = float64(preferenceResult.Confidence)
	}

	logging.Debugf("[Signal Computation] Preference signal evaluation completed in %v", elapsed)
	if err != nil {
		if errors.Is(err, ErrPreferenceBelowThreshold) {
			logging.Debugf("preference rule did not match threshold: %v", err)
			return
		}
		logging.Errorf("preference rule evaluation failed: %v", err)
		return
	}
	if preferenceResult == nil || preferenceName == "" || !c.hasConfiguredPreferenceRule(preferenceName) {
		return
	}

	recordPreferenceMatchMetrics(preferenceName)
	details := c.contrastivePreferenceDetails(text)

	mu.Lock()
	recordPreferenceSignalValues(results, preferenceResult, details)
	mu.Unlock()

	logging.Debugf("Preference rule matched: %s", preferenceName)
}

func (c *Classifier) hasConfiguredPreferenceRule(name string) bool {
	for _, rule := range c.Config.PreferenceRules {
		if rule.Name == name {
			return true
		}
	}
	return false
}

func (c *Classifier) contrastivePreferenceDetails(text string) *PreferenceClassificationDetails {
	if c.preferenceClassifier == nil || !c.preferenceClassifier.useContrastive || c.preferenceClassifier.contrastive == nil {
		return nil
	}
	details, err := c.preferenceClassifier.contrastive.ClassifyDetailed(text)
	if err != nil {
		return nil
	}
	return details
}

func recordPreferenceSignalValues(
	results *SignalResults,
	preferenceResult *PreferenceResult,
	details *PreferenceClassificationDetails,
) {
	if preferenceResult == nil {
		return
	}
	key := "preference:" + preferenceResult.Preference
	results.MatchedPreferenceRules = append(results.MatchedPreferenceRules, preferenceResult.Preference)
	results.SignalConfidences[key] = float64(preferenceResult.Confidence)
	results.SignalValues[key] = float64(preferenceResult.Confidence)
	results.SignalValues[key+":margin"] = float64(preferenceResult.Margin)
	recordPreferenceDetailValues(results, details)
}

func recordPreferenceDetailValues(results *SignalResults, details *PreferenceClassificationDetails) {
	if details == nil {
		return
	}
	for _, score := range details.Scores {
		results.SignalValues["preference:"+score.Name] = float64(score.Score)
		results.SignalValues["preference:"+score.Name+":best"] = float64(score.Best)
		results.SignalValues["preference:"+score.Name+":support"] = float64(score.Support)
		results.SignalValues["preference:"+score.Name+":prototype_count"] = float64(score.PrototypeCount)
	}
}

func recordPreferenceMatchMetrics(preferenceName string) {
	metrics.RecordSignalMatch(config.SignalTypePreference, preferenceName)
}
