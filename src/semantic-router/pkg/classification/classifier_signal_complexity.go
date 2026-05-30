package classification

import (
	"fmt"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func (c *Classifier) evaluateComplexitySignal(results *SignalResults, mu *sync.Mutex, text string, imageURL string, imgCache *requestImageEmbeddingCache) {
	start := time.Now()
	classifyResults, err := c.complexityClassifier.classifyDetailedWithImageCached(text, imageURL, imgCache)
	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()

	// Record signal extraction metrics for each matched rule
	results.Metrics.Complexity.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0

	logging.Debugf("[Signal Computation] Complexity signal evaluation completed in %v", elapsed)
	if err != nil {
		logging.Errorf("complexity rule evaluation failed: %v", err)
		return
	}

	bestConfidence := 0.0
	mu.Lock()
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
	results.Metrics.Complexity.Confidence = bestConfidence
	mu.Unlock()
}
