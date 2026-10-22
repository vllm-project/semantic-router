package classification

import (
	"context"
	"errors"
	"maps"
	"math"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
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
	var rules []config.ClassifierSignalRule
	for _, rule := range c.Config.ClassifierRules {
		if signalRuleUsed(usedSignals, config.SignalTypeClassifier, rule.Name) && c.genericClassifiers[rule.Name] != nil {
			rules = append(rules, rule)
		}
	}
	modelservice.Fan(ctx, len(rules), func(i int) {
		c.evaluateGenericClassifierRule(ctx, results, mu, text, rules[i], c.genericClassifiers[rules[i].Name])
	})
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
	c.recordSignalExtraction(
		config.SignalTypeClassifier,
		rule.Name,
		time.Since(start).Seconds(),
	)
	if err != nil {
		code := genericClassifierError(err)
		mu.Lock()
		results.SignalErrors[signalConfidenceKey(config.SignalTypeClassifier, rule.Name)] = code
		appendClassifierObservation(results, rule, nil, nil, code)
		mu.Unlock()
		return
	}
	if !classifierScoresFinite(rule.Labels, result.Scores) || (result.Thresholds != nil && !classifierScoresFinite(rule.Labels, result.Thresholds)) {
		mu.Lock()
		results.SignalErrors[signalConfidenceKey(config.SignalTypeClassifier, rule.Name)] = genericClassifierInvalidScoreCode
		appendClassifierObservation(results, rule, nil, nil, genericClassifierInvalidScoreCode)
		mu.Unlock()
		return
	}
	mu.Lock()
	defer mu.Unlock()
	if result.PolicyTrace != nil {
		result.PolicyTrace.ExecutionTimeMs = float64(time.Since(start).Microseconds()) / 1000
		if results.Metrics.Classifier.Rules == nil {
			results.Metrics.Classifier.Rules = make(map[string]*ClassifierRuleMetrics)
		}
		results.Metrics.Classifier.Rules[rule.Name] = result.PolicyTrace
	}
	bestLabel := ""
	bestLabelScore := -1.0
	matchedLabels := make([]string, 0, len(rule.Labels))
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
		if result.Thresholds != nil && score >= result.Thresholds[label] {
			matchedLabels = append(matchedLabels, label)
			labelMatch := rule.Name + ":" + label
			results.MatchedClassifierRules = append(results.MatchedClassifierRules, labelMatch)
			c.recordSignalMatch(config.SignalTypeClassifier, labelMatch)
		}
	}
	if result.Thresholds == nil && bestLabel != "" {
		matchedLabels = append(matchedLabels, bestLabel)
		labelMatch := rule.Name + ":" + bestLabel
		results.MatchedClassifierRules = append(
			results.MatchedClassifierRules,
			labelMatch,
		)
		c.recordSignalMatch(config.SignalTypeClassifier, labelMatch)
	}
	appendClassifierObservation(results, rule, &result, matchedLabels, "")
}

func appendClassifierObservation(
	results *SignalResults,
	rule config.ClassifierSignalRule,
	result *labelClassification,
	matchedLabels []string,
	errorCode string,
) {
	observation := decision.SignalObservation{
		Version:      decision.SignalObservationVersionV1,
		SignalType:   config.SignalTypeClassifier,
		Name:         rule.Name,
		Availability: decision.SignalObservationUnavailable,
		Provenance: decision.SignalObservationProvenance{
			Evaluator: config.SignalTypeClassifier,
			Backend:   rule.Type,
		},
		PayloadType: config.RemoteClassifierContractLabelDistribution,
		ErrorCode:   errorCode,
	}
	if result != nil {
		observation.Availability = decision.SignalObservationAvailable
		observation.Payload = &decision.LabelDistributionObservation{
			Scores:        maps.Clone(result.Scores),
			Thresholds:    maps.Clone(result.Thresholds),
			MatchedLabels: append([]string(nil), matchedLabels...),
			Rationale:     result.Rationale,
		}
	}
	results.Observations = append(results.Observations, observation)
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
	if len(labels) == 0 || len(scores) != len(labels) {
		return false
	}
	for _, label := range labels {
		score, exists := scores[label]
		if !exists || math.IsNaN(score) || math.IsInf(score, 0) || score < 0 || score > 1 {
			return false
		}
	}
	return true
}

func classifierLabelKey(name string, label string) string {
	return config.SignalTypeClassifier + ":" + name + ":" + label
}
