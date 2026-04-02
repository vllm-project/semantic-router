package classification

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

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
