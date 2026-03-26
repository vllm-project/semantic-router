package classification

import (
	"slices"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// evaluateKBSignals runs all configured knowledge bases, records structured KB
// results, and maps routing.signals.kb bindings into normal matched signals.
func (c *Classifier) evaluateKBSignals(results *SignalResults, mu *sync.Mutex, text string) {
	if len(c.kbClassifiers) == 0 {
		return
	}

	start := time.Now()
	classifierResults := make(map[string]*KBClassifyResult, len(c.kbClassifiers))
	for name, classifier := range c.kbClassifiers {
		classifyResult, err := classifier.Classify(text)
		if err != nil {
			logging.Warnf("[KB Signal] KB %q failed: %v", name, err)
			continue
		}
		classifierResults[name] = classifyResult
	}

	mu.Lock()
	defer mu.Unlock()

	if results.KBClassifierResults == nil {
		results.KBClassifierResults = make(map[string]*KBClassifyResult, len(classifierResults))
	}
	if results.KBMetricValues == nil {
		results.KBMetricValues = make(map[string]float64, len(classifierResults))
	}

	for kbName, classifyResult := range classifierResults {
		results.KBClassifierResults[kbName] = classifyResult
		for metricName, value := range classifyResult.MetricValues {
			metricKey := kbMetricKey(kbName, metricName)
			results.KBMetricValues[metricKey] = value
			results.SignalValues[metricKey] = value
		}
	}

	for _, rule := range c.Config.KBRules {
		classifyResult, ok := classifierResults[rule.KB]
		if !ok {
			continue
		}
		confidence, matched := kbSignalMatchConfidence(rule, classifyResult)
		if !matched {
			continue
		}
		results.MatchedKBRules = append(results.MatchedKBRules, rule.Name)
		results.SignalConfidences[config.SignalTypeKB+":"+rule.Name] = confidence
		metrics.RecordSignalMatch(config.SignalTypeKB, rule.Name)
	}

	results.Metrics.KB.ExecutionTimeMs = float64(time.Since(start).Microseconds()) / 1000.0
	results.Metrics.KB.Confidence = signalSetBestConfidence(results.SignalConfidences, config.SignalTypeKB, results.MatchedKBRules)
	metrics.RecordSignalExtraction(config.SignalTypeKB, "kb", time.Since(start).Seconds())
}

func kbSignalMatchConfidence(rule config.KBSignalRule, result *KBClassifyResult) (float64, bool) {
	matchMode := kbSignalMatchMode(rule)
	switch rule.Target.Kind {
	case config.KBTargetKindLabel:
		return kbLabelMatchConfidence(rule.Target.Value, matchMode, result)
	case config.KBTargetKindGroup:
		return kbGroupMatchConfidence(rule.Target.Value, matchMode, result)
	}
	return 0, false
}

func kbSignalMatchMode(rule config.KBSignalRule) string {
	if rule.Match == "" {
		return config.KBMatchThreshold
	}
	return rule.Match
}

func kbLabelMatchConfidence(labelName string, matchMode string, result *KBClassifyResult) (float64, bool) {
	switch matchMode {
	case config.KBMatchBest:
		if result.BestLabel == labelName && result.BestLabel != "" {
			return result.BestSimilarity, true
		}
		return 0, false
	default:
		confidence, ok := result.LabelConfidences[labelName]
		if !ok || !slices.Contains(result.MatchedLabels, labelName) {
			return 0, false
		}
		return confidence, true
	}
}

func kbGroupMatchConfidence(groupName string, matchMode string, result *KBClassifyResult) (float64, bool) {
	confidence, ok := result.GroupScores[groupName]
	if !ok {
		return 0, false
	}
	switch matchMode {
	case config.KBMatchBest:
		if result.BestGroup == groupName && result.BestGroup != "" {
			return confidence, true
		}
		return 0, false
	default:
		if slices.Contains(result.MatchedGroups, groupName) {
			return confidence, true
		}
		return 0, false
	}
}

func signalSetBestConfidence(confidences map[string]float64, signalType string, names []string) float64 {
	best := 0.0
	for _, name := range names {
		if confidence := confidences[signalType+":"+name]; confidence > best {
			best = confidence
		}
	}
	return best
}
