package classification

import (
	"context"
	"errors"
	"math"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

const (
	genericClassifierErrorCode        = "classifier_evaluation_failed"
	genericClassifierInvalidScoreCode = "classifier_invalid_score"
	genericClassifierCancelledCode    = "classifier_evaluation_cancelled"
	genericClassifierTimeoutCode      = "classifier_evaluation_timeout"
)

func (c *Classifier) evaluateGenericClassifierSignals(
	results *SignalResults,
	mu *sync.Mutex,
	text string,
	usedSignals map[string]bool,
	ctx context.Context,
) {
	start := time.Now()
	if ctx == nil {
		ctx = context.Background()
	}
	var waitGroup sync.WaitGroup
	for _, rule := range c.Config.ClassifierRules {
		if !signalRuleUsed(usedSignals, config.SignalTypeClassifier, rule.Name) {
			continue
		}
		classifier := c.genericClassifiers[rule.Name]
		if classifier == nil {
			continue
		}
		waitGroup.Add(1)
		go func(rule config.ClassifierSignalRule, classifier labelClassifier) {
			defer waitGroup.Done()
			c.evaluateGenericClassifierRule(
				ctx,
				results,
				mu,
				text,
				rule,
				classifier,
			)
		}(rule, classifier)
	}
	waitGroup.Wait()
	elapsed := time.Since(start)
	mu.Lock()
	results.Metrics.Classifier.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	mu.Unlock()
}

func (c *Classifier) evaluateGenericClassifierRule(
	ctx context.Context,
	results *SignalResults,
	mu *sync.Mutex,
	text string,
	rule config.ClassifierSignalRule,
	classifier labelClassifier,
) {
	start := time.Now()
	result, err := classifier.Classify(ctx, text)
	metrics.RecordSignalExtraction(
		config.SignalTypeClassifier,
		rule.Name,
		time.Since(start).Seconds(),
	)
	if err != nil {
		mu.Lock()
		results.SignalErrors[signalConfidenceKey(
			config.SignalTypeClassifier,
			rule.Name,
		)] = genericClassifierError(err)
		mu.Unlock()
		return
	}
	if !classifierScoresFinite(rule.Labels, result.Scores) {
		mu.Lock()
		results.SignalErrors[signalConfidenceKey(
			config.SignalTypeClassifier,
			rule.Name,
		)] = genericClassifierInvalidScoreCode
		mu.Unlock()
		return
	}
	mu.Lock()
	defer mu.Unlock()
	bestLabel := ""
	bestLabelScore := -1.0
	for _, label := range rule.Labels {
		score := result.Scores[label]
		key := classifierLabelKey(rule.Name, label)
		results.SignalValues[key] = score
		results.SignalConfidences[key] = score
		if score > results.Metrics.Classifier.Confidence {
			results.Metrics.Classifier.Confidence = score
		}
		if score > bestLabelScore {
			bestLabel = label
			bestLabelScore = score
		}
	}
	if bestLabel != "" {
		labelMatch := rule.Name + ":" + bestLabel
		results.MatchedClassifierRules = append(
			results.MatchedClassifierRules,
			labelMatch,
		)
		metrics.RecordSignalMatch(config.SignalTypeClassifier, labelMatch)
	}
}

func genericClassifierError(err error) string {
	if errors.Is(err, context.Canceled) {
		return genericClassifierCancelledCode
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return genericClassifierTimeoutCode
	}
	return genericClassifierErrorCode
}

func classifierScoresFinite(
	labels []string,
	scores map[string]float64,
) bool {
	for _, label := range labels {
		score := scores[label]
		if math.IsNaN(score) || math.IsInf(score, 0) {
			return false
		}
	}
	return true
}

func classifierLabelKey(name string, label string) string {
	return config.SignalTypeClassifier + ":" + name + ":" + label
}
