package classification

import (
	"context"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

const genericClassifierErrorCode = "classifier_evaluation_failed"

func (c *Classifier) evaluateGenericClassifierSignals(
	results *SignalResults,
	mu *sync.Mutex,
	text string,
	usedSignals map[string]bool,
) {
	start := time.Now()
	bestConfidence := 0.0
	for _, rule := range c.Config.ClassifierRules {
		if !signalRuleUsed(usedSignals, config.SignalTypeClassifier, rule.Name) {
			continue
		}
		classifier := c.genericClassifiers[rule.Name]
		if classifier == nil {
			continue
		}
		result, err := classifier.Classify(context.Background(), text)
		if err != nil {
			mu.Lock()
			results.SignalErrors[signalConfidenceKey(
				config.SignalTypeClassifier,
				rule.Name,
			)] = genericClassifierErrorCode
			mu.Unlock()
			continue
		}
		mu.Lock()
		bestLabel := ""
		bestLabelScore := -1.0
		for _, label := range rule.Labels {
			score := result.Scores[label]
			key := classifierLabelKey(rule.Name, label)
			results.SignalValues[key] = score
			results.SignalConfidences[key] = score
			if score > bestConfidence {
				bestConfidence = score
			}
			if score > bestLabelScore {
				bestLabel = label
				bestLabelScore = score
			}
		}
		if bestLabel != "" {
			labelMatch := rule.Name + ":" + bestLabel
			results.MatchedClassifierRules = append(results.MatchedClassifierRules, labelMatch)
			metrics.RecordSignalMatch(config.SignalTypeClassifier, labelMatch)
		}
		mu.Unlock()
	}
	elapsed := time.Since(start)
	results.Metrics.Classifier.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	results.Metrics.Classifier.Confidence = bestConfidence
}

func classifierLabelKey(name string, label string) string {
	return config.SignalTypeClassifier + ":" + name + ":" + label
}
