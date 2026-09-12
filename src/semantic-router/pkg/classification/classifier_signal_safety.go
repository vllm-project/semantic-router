package classification

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type safetyDetector struct {
	binary    labelClassifier
	hazard    labelClassifier
	binaryKey string
	hazardKey string
}

// A recipe owns one instance per model contract, even when several rules use
// different thresholds over that head. Cleanup is safe through every consumer.
type sharedSafetyHead struct {
	labelClassifier
	closeOnce sync.Once
	closeErr  error
}

func (s *sharedSafetyHead) Close() error {
	s.closeOnce.Do(func() {
		if closer, ok := s.labelClassifier.(interface{ Close() error }); ok {
			s.closeErr = closer.Close()
		}
	})
	return s.closeErr
}

func (s *sharedSafetyHead) Initialize() error {
	if initializer, ok := s.labelClassifier.(interface{ Initialize() error }); ok {
		return initializer.Initialize()
	}
	return nil
}

func (s *safetyDetector) Close() error {
	var failures []error
	for _, classifier := range []labelClassifier{s.binary, s.hazard} {
		if closer, ok := classifier.(interface{ Close() error }); ok {
			failures = append(failures, closer.Close())
		}
	}
	return errors.Join(failures...)
}

// Preparation owns model resolution and cleanup; request evaluation only calls
// the prepared full-distribution classifier interface.
func (b *classifierOptionBuilder) buildSafetyClassifiersOption() (option, error) {
	models := make(map[string]*safetyDetector, len(b.cfg.SafetyRules))
	heads := make(map[string]*sharedSafetyHead)
	prepare := func(consumer, model string, labels []string, multiLabel bool) (labelClassifier, string, error) {
		key := b.safetyModelKey(model, labels, multiLabel)
		if existing := heads[key]; existing != nil {
			return existing, key, nil
		}
		prepared, err := b.prepareSafetyClassifier(consumer, model, labels, multiLabel)
		if err != nil {
			return nil, key, err
		}
		shared := &sharedSafetyHead{labelClassifier: prepared}
		heads[key] = shared
		return shared, key, nil
	}
	for _, rule := range b.cfg.SafetyRules {
		detector := &safetyDetector{}
		models[rule.Name] = detector
		var err error
		detector.binary, detector.binaryKey, err = prepare("safety."+rule.Name, rule.Model, rule.EffectiveLabels(), false)
		if err == nil && rule.Hazard != nil {
			detector.hazard, detector.hazardKey, err = prepare("safety."+rule.Name+".hazard", rule.Hazard.Model, rule.Hazard.Labels, true)
		}
		if err != nil {
			for _, prepared := range models {
				_ = prepared.Close()
			}
			return nil, fmt.Errorf("prepare safety signal %q: %w", rule.Name, err)
		}
	}
	return func(c *Classifier) { c.safetyClassifiers = models }, nil
}

func (b *classifierOptionBuilder) prepareSafetyClassifier(consumer, model string, labels []string, multiLabel bool) (labelClassifier, error) {
	if model == "" {
		local := b.cfg.SafetyModels.Safety
		if multiLabel {
			local = b.cfg.SafetyModels.Hazard
		}
		return newNativeSafetyClassifier(local, labels, multiLabel)
	}
	rule := config.ClassifierSignalRule{Name: consumer, Type: config.ClassifierSignalTypeSequenceClassifier, Model: model, Labels: labels}
	classifier, err := newSequenceLabelClassifier(rule, b.cfg.FindExternalModelByName(model))
	if err == nil && multiLabel {
		classifier.(*sequenceLabelClassifier).backend.multiLabel = true
	}
	return classifier, err
}

func (b *classifierOptionBuilder) safetyModelKey(model string, labels []string, multiLabel bool) string {
	local := config.SequenceHeadModelConfig{}
	if model == "" {
		local = b.cfg.SafetyModels.Safety
		if multiLabel {
			local = b.cfg.SafetyModels.Hazard
		}
	}
	// Identical model/label contracts can share one call within this request.
	// Consumer identity and the configured thresholds remain separate.
	key, _ := json.Marshal(struct {
		Model      string
		Local      config.SequenceHeadModelConfig
		Labels     []string
		MultiLabel bool
	}{model, local, labels, multiLabel})
	return string(key)
}

func (c *Classifier) initializeSafetyClassifiers() error {
	for _, rule := range c.Config.SafetyRules {
		detector := c.safetyClassifiers[rule.Name]
		if detector == nil {
			return fmt.Errorf("safety classifier %q is unavailable", rule.Name)
		}
		for _, head := range []labelClassifier{detector.binary, detector.hazard} {
			if initializer, ok := head.(interface{ Initialize() error }); ok {
				if err := initializer.Initialize(); err != nil {
					return fmt.Errorf("initialize safety %q: %w", rule.Name, err)
				}
			}
		}
	}
	return nil
}

type safetyCachedResult struct {
	result labelClassification
	err    error
}

func (c *Classifier) evaluateSafetySignals(ctx context.Context, results *SignalResults, mu *sync.Mutex, text string, used map[string]bool) {
	if ctx == nil {
		ctx = context.Background()
	}
	start := time.Now()
	cache := make(map[string]safetyCachedResult)
	classify := func(key string, classifier labelClassifier) (labelClassification, error) {
		if previous, ok := cache[key]; ok {
			return previous.result, previous.err
		}
		if classifier == nil {
			return labelClassification{}, fmt.Errorf("safety head is unavailable")
		}
		result, err := classifier.Classify(ctx, text)
		cache[key] = safetyCachedResult{result, err}
		return result, err
	}
	for _, rule := range c.Config.SafetyRules {
		if !signalRuleUsed(used, config.SignalTypeSafety, rule.Name) {
			continue
		}
		detector := c.safetyClassifiers[rule.Name]
		key := config.SignalTypeSafety + ":" + rule.Name
		if detector == nil {
			mu.Lock()
			results.SignalErrors[key] = "safety_model_unavailable"
			mu.Unlock()
			continue
		}
		began := time.Now()
		binary, err := classify(detector.binaryKey, detector.binary)
		c.recordSignalExtraction(config.SignalTypeSafety, rule.Name, time.Since(began).Seconds())
		if err != nil || !classifierScoresFinite(rule.EffectiveLabels(), binary.Scores) {
			mu.Lock()
			results.SignalErrors[key] = "safety_classification_failed"
			mu.Unlock()
			continue
		}
		risk := selectedSafetyScore(binary.Scores, rule.EffectiveUnsafeLabels())
		matched := risk >= rule.Threshold
		mu.Lock()
		results.SignalValues[key] = risk
		results.SignalConfidences[key] = risk
		results.Metrics.Safety.Confidence = max(results.Metrics.Safety.Confidence, risk)
		mu.Unlock()
		if matched && rule.Hazard != nil {
			hazard, err := classify(detector.hazardKey, detector.hazard)
			if err != nil || !classifierScoresFinite(rule.Hazard.Labels, hazard.Scores) {
				mu.Lock()
				results.SignalErrors[key] = "safety_hazard_classification_failed"
				mu.Unlock()
				continue
			}
			hazardRisk := selectedHazardScore(hazard.Scores, rule.Hazard.Categories)
			matched = hazardRisk >= rule.Hazard.Threshold
			mu.Lock()
			results.SignalValues[key+":hazard"] = hazardRisk
			mu.Unlock()
		}
		if matched {
			mu.Lock()
			results.MatchedSafetyRules = append(results.MatchedSafetyRules, rule.Name)
			mu.Unlock()
			c.recordSignalMatch(config.SignalTypeSafety, rule.Name)
		}
	}
	mu.Lock()
	results.Metrics.Safety.ExecutionTimeMs = float64(time.Since(start).Microseconds()) / 1000
	mu.Unlock()
}

func selectedSafetyScore(scores map[string]float64, labels []string) float64 {
	var result float64
	for _, label := range labels {
		result += scores[label]
	}
	return result
}

// A category rule matches if any selected hazard exceeds its threshold.
// Independent sigmoid probabilities cannot be added as disjoint mass.
func selectedHazardScore(scores map[string]float64, labels []string) float64 {
	var result float64
	for _, label := range labels {
		result = max(result, scores[label])
	}
	return result
}
