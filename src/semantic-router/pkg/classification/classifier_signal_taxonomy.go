package classification

import (
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// evaluateTaxonomySignals runs all configured taxonomy classifiers, records
// structured classifier results, and maps routing.signals.taxonomy bindings
// into normal matched signals.
func (c *Classifier) evaluateTaxonomySignals(results *SignalResults, mu *sync.Mutex, text string) {
	if len(c.taxonomyClassifiers) == 0 {
		return
	}

	start := time.Now()
	classifierResults := make(map[string]*TaxonomyClassifyResult, len(c.taxonomyClassifiers))
	for name, classifier := range c.taxonomyClassifiers {
		classifyResult, err := classifier.Classify(text)
		if err != nil {
			logging.Warnf("[Taxonomy Signal] classifier %q failed: %v", name, err)
			continue
		}
		classifierResults[name] = classifyResult
	}

	mu.Lock()
	defer mu.Unlock()

	if results.TaxonomyClassifierResults == nil {
		results.TaxonomyClassifierResults = make(map[string]*TaxonomyClassifyResult, len(classifierResults))
	}
	if results.TaxonomyMetricValues == nil {
		results.TaxonomyMetricValues = make(map[string]float64, len(classifierResults))
	}

	for classifierName, classifyResult := range classifierResults {
		results.TaxonomyClassifierResults[classifierName] = classifyResult
		metricKey := taxonomyMetricKey(classifierName, config.TaxonomyMetricContrastive)
		results.TaxonomyMetricValues[metricKey] = classifyResult.ContrastiveScore
		results.SignalValues[metricKey] = classifyResult.ContrastiveScore
	}

	for _, rule := range c.Config.TaxonomyRules {
		classifyResult, ok := classifierResults[rule.Classifier]
		if !ok {
			continue
		}
		confidence, matched := taxonomySignalMatchConfidence(rule, classifyResult)
		if !matched {
			continue
		}
		results.MatchedTaxonomyRules = append(results.MatchedTaxonomyRules, rule.Name)
		results.SignalConfidences[config.SignalTypeTaxonomy+":"+rule.Name] = confidence
		metrics.RecordSignalMatch(config.SignalTypeTaxonomy, rule.Name)
	}

	results.Metrics.Taxonomy.ExecutionTimeMs = float64(time.Since(start).Microseconds()) / 1000.0
	results.Metrics.Taxonomy.Confidence = signalSetBestConfidence(results.SignalConfidences, config.SignalTypeTaxonomy, results.MatchedTaxonomyRules)
	metrics.RecordSignalExtraction(config.SignalTypeTaxonomy, "taxonomy", time.Since(start).Seconds())
}

func taxonomySignalMatchConfidence(rule config.TaxonomySignalRule, result *TaxonomyClassifyResult) (float64, bool) {
	switch rule.Bind.Kind {
	case config.TaxonomyBindKindTier:
		if result.BestMatchedTier == rule.Bind.Value && result.BestMatchedCategory != "" {
			return result.BestMatchedSimilarity, true
		}
	case config.TaxonomyBindKindCategory:
		if confidence, ok := result.CategoryConfidences[rule.Bind.Value]; ok {
			for _, category := range result.MatchedCategories {
				if category == rule.Bind.Value {
					return confidence, true
				}
			}
		}
	}
	return 0, false
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
