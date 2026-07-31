package classification

import (
	"fmt"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// evaluateComplexitySignal evaluates all complexity rules. Embedding-mode rules use the
// prototype-bank similarity classifier; rules with method: model use the trained complexity
// classifier. Both modes emit the same "rule:difficulty" matches so downstream consumers
// (projections, decisions) stay mode-agnostic.
func (c *Classifier) evaluateComplexitySignal(results *SignalResults, mu *sync.Mutex, text string, imageURL string, imgCache *requestImageEmbeddingCache) {
	start := time.Now()

	bestConfidence := 0.0
	if c.complexityClassifier != nil {
		if conf := c.evaluateEmbeddingComplexityRules(results, mu, text, imageURL, imgCache, start); conf > bestConfidence {
			bestConfidence = conf
		}
	}
	if c.complexityModelInference != nil && c.ComplexityMapping != nil {
		if conf := c.evaluateModelComplexityRules(results, mu, text, start); conf > bestConfidence {
			bestConfidence = conf
		}
	}

	elapsed := time.Since(start)
	results.Metrics.Complexity.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	mu.Lock()
	results.Metrics.Complexity.Confidence = bestConfidence
	mu.Unlock()
	logging.Debugf("[Signal Computation] Complexity signal evaluation completed in %v", elapsed)
}

// evaluateEmbeddingComplexityRules runs the prototype-bank similarity classifier over the
// embedding-mode complexity rules and records the matched difficulties. It returns the
// highest match confidence observed.
func (c *Classifier) evaluateEmbeddingComplexityRules(results *SignalResults, mu *sync.Mutex, text string, imageURL string, imgCache *requestImageEmbeddingCache, start time.Time) float64 {
	classifyResults, err := c.complexityClassifier.classifyDetailedWithImageCached(text, imageURL, imgCache)
	if err != nil {
		logging.Errorf("complexity rule evaluation failed: %v", err)
		return 0
	}

	latencySeconds := time.Since(start).Seconds()
	bestConfidence := 0.0
	mu.Lock()
	defer mu.Unlock()
	for _, result := range classifyResults {
		matchName := fmt.Sprintf("%s:%s", result.RuleName, result.Difficulty)
		metrics.RecordSignalExtraction(config.SignalTypeComplexity, matchName, latencySeconds)
		metrics.RecordSignalMatch(config.SignalTypeComplexity, matchName)
		results.MatchedComplexityRules = append(results.MatchedComplexityRules, matchName)
		results.SignalConfidences["complexity:"+matchName] = result.Confidence
		results.SignalValues["complexity:"+result.RuleName+":text_hard_score"] = result.TextHardScore
		results.SignalValues["complexity:"+result.RuleName+":text_easy_score"] = result.TextEasyScore
		results.SignalValues["complexity:"+result.RuleName+":text_margin"] = result.TextMargin
		results.SignalValues["complexity:"+result.RuleName+":image_hard_score"] = result.ImageHardScore
		results.SignalValues["complexity:"+result.RuleName+":image_easy_score"] = result.ImageEasyScore
		results.SignalValues["complexity:"+result.RuleName+":image_margin"] = result.ImageMargin
		results.SignalValues["complexity:"+result.RuleName+":margin"] = result.FusedMargin
		if result.Confidence > bestConfidence {
			bestConfidence = result.Confidence
		}
	}
	return bestConfidence
}

// evaluateModelComplexityRules runs the trained complexity classifier once and emits a
// difficulty match for every complexity rule with method: model. The classifier is text-only.
// rule.Threshold acts as a confidence floor: when the top-class confidence is below it, the
// rule reports the neutral "medium" band (matching the embedding path's neutral semantics).
// It returns the highest match confidence observed.
func (c *Classifier) evaluateModelComplexityRules(results *SignalResults, mu *sync.Mutex, text string, start time.Time) float64 {
	modelRules := make([]config.ComplexityRule, 0, len(c.Config.ComplexityRules))
	for _, rule := range c.Config.ComplexityRules {
		if rule.UsesModel() {
			modelRules = append(modelRules, rule)
		}
	}
	if len(modelRules) == 0 {
		return 0
	}

	classResult, err := c.complexityModelInference.Classify(text)
	if err != nil {
		logging.Errorf("complexity model classification failed: %v", err)
		return 0
	}
	difficulty, ok := c.ComplexityMapping.GetDifficultyFromIndex(classResult.Class)
	if !ok {
		logging.Errorf("complexity model returned unmapped class index %d", classResult.Class)
		return 0
	}
	confidence := float64(classResult.Confidence)
	latencySeconds := time.Since(start).Seconds()

	bestConfidence := 0.0
	mu.Lock()
	defer mu.Unlock()
	for _, rule := range modelRules {
		ruleDifficulty := difficulty
		// Confidence floor: below the rule threshold, report the neutral band.
		if confidence < float64(rule.Threshold) {
			ruleDifficulty = "medium"
		}
		matchName := fmt.Sprintf("%s:%s", rule.Name, ruleDifficulty)
		metrics.RecordSignalExtraction(config.SignalTypeComplexity, matchName, latencySeconds)
		metrics.RecordSignalMatch(config.SignalTypeComplexity, matchName)
		results.MatchedComplexityRules = append(results.MatchedComplexityRules, matchName)
		results.SignalConfidences["complexity:"+matchName] = confidence
		results.SignalValues["complexity:"+rule.Name+":margin"] = signedComplexityMargin(ruleDifficulty, confidence)
		if confidence > bestConfidence {
			bestConfidence = confidence
		}
	}
	return bestConfidence
}

// signedComplexityMargin maps a difficulty label and confidence to a signed margin so that
// projection inputs reading the complexity ":margin" value behave consistently across modes:
// positive for hard, negative for easy, zero for the neutral medium band.
func signedComplexityMargin(difficulty string, confidence float64) float64 {
	switch difficulty {
	case "hard":
		return confidence
	case "easy":
		return -confidence
	default:
		return 0
	}
}
