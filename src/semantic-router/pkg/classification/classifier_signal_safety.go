package classification

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
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
	runtime := b.models
	if runtime == nil {
		var err error
		runtime, err = newClassifierModelRuntime(b.cfg, nil)
		if err != nil {
			return nil, err
		}
	}
	prepare := func(consumer, model string, labels []string, multiLabel bool) (labelClassifier, string, error) {
		spec, window, err := b.safetySpec(runtime, consumer, model, multiLabel)
		if err != nil {
			return nil, "", err
		}
		key := safetyModelKey(spec, window, labels)
		prepared, err := b.prepareSafetyClassifier(runtime, spec, window, labels, multiLabel)
		return prepared, key, err
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

func (b *classifierOptionBuilder) safetySpec(models *classifierModelRuntime, consumer, model string, multiLabel bool) (config.ResolvedModelBinding, *config.SequenceHeadWindowConfig, error) {
	local := b.cfg.SafetyModels.Safety
	contract := config.RemoteClassifierContractLabelDistribution
	if multiLabel {
		local = b.cfg.SafetyModels.Hazard
		contract = config.RemoteClassifierContractLabelScores
	}
	var spec config.ResolvedModelBinding
	if model != "" {
		spec = models.remoteSpec(consumer, &config.RemoteClassifierBackend{Model: model, Protocol: config.RemoteClassifierProtocolHTTPClassify, Contract: contract})
	} else {
		spec = models.localSpec(consumer, local.ModelID, "modernbert", contract, local.UseCPU, local.MaxSequenceLength)
		if _, declared := models.plan.Lookup(models.recipe, consumer); !declared {
			deployment := "safety"
			if multiLabel {
				deployment = "hazard"
			}
			spec.Binding.Deployment = deployment
			spec.Admission = b.cfg.ModelAdmission[deployment]
		}
	}
	if spec.Binding.Contract != contract {
		return spec, nil, fmt.Errorf("safety head has incompatible result contract")
	}
	var window *config.SequenceHeadWindowConfig
	if model == "" && local.Window != nil {
		copy := *local.Window
		window = &copy
	}
	if spec.Deployment.Provider == "http" && window != nil {
		return spec, nil, fmt.Errorf("remote safety head cannot use local token windows")
	}
	return spec, window, nil
}

func (b *classifierOptionBuilder) prepareSafetyClassifier(models *classifierModelRuntime, spec config.ResolvedModelBinding, window *config.SequenceHeadWindowConfig, labels []string, multiLabel bool) (labelClassifier, error) {
	if spec.Deployment.Provider == "http" {
		external, err := config.ResolveRemoteClassifierBackend(b.cfg, &config.RemoteClassifierBackend{Model: spec.Deployment.ExternalModel, Protocol: spec.Binding.Adapter, Contract: spec.Binding.Contract}, config.ModelRoleClassification, spec.Binding.Contract)
		if err != nil {
			return nil, err
		}
		return prepareRemoteSafetyClassifier(models, spec, external, labels, multiLabel)
	}
	return newOwnedSafetyClassifier(models, spec, labels, multiLabel, window), nil
}

func safetyModelKey(spec config.ResolvedModelBinding, window *config.SequenceHeadWindowConfig, labels []string) string {
	// Share request results only for the same effective execution and vector
	// contract; each consumer still owns a separate typed binding handle.
	computation := spec.Binding
	computation.Deployment = "" // Catalog/consumer names do not change a forward pass.
	key, _ := json.Marshal(struct {
		Deployment config.ModelDeployment
		Binding    config.ModelBinding
		Window     *config.SequenceHeadWindowConfig
		Labels     []string
	}{spec.Deployment.WithDefaults(), computation, window, labels})
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
		if err != nil || !safetyScoresFinite(rule.EffectiveLabels(), binary) {
			mu.Lock()
			results.SignalErrors[key] = "safety_classification_failed"
			mu.Unlock()
			continue
		}
		risk := aggregateSafetyWindows(binary, rule.EffectiveUnsafeLabels(), selectedSafetyScore)
		matched := risk >= rule.Threshold
		mu.Lock()
		results.SignalValues[key] = risk
		results.SignalConfidences[key] = risk
		results.Metrics.Safety.Confidence = max(results.Metrics.Safety.Confidence, risk)
		mu.Unlock()
		if matched && rule.Hazard != nil {
			hazard, err := classify(detector.hazardKey, detector.hazard)
			if err != nil || !safetyScoresFinite(rule.Hazard.Labels, hazard) {
				mu.Lock()
				results.SignalErrors[key] = "safety_hazard_classification_failed"
				mu.Unlock()
				continue
			}
			hazardRisk := aggregateSafetyWindows(hazard, rule.Hazard.Categories, selectedHazardScore)
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

func safetyScoresFinite(labels []string, result labelClassification) bool {
	windows := result.ScoreWindows
	if len(windows) == 0 {
		windows = []map[string]float64{result.Scores}
	}
	for _, scores := range windows {
		if len(scores) != len(labels) {
			return false
		}
		for _, label := range labels {
			value, ok := scores[label]
			if !ok || math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || value > 1 {
				return false
			}
		}
	}
	return true
}

// Select the rule's labels within each window, then take the maximum. Adding
// independently maximized softmax classes would invent probability mass.
func aggregateSafetyWindows(result labelClassification, labels []string, selectScore func(map[string]float64, []string) float64) float64 {
	if len(result.ScoreWindows) == 0 {
		return selectScore(result.Scores, labels)
	}
	var maximum float64
	for _, scores := range result.ScoreWindows {
		maximum = max(maximum, selectScore(scores, labels))
	}
	return maximum
}
