package classification

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func (c *Classifier) evaluateComplexitySignal(ctx context.Context, results *SignalResults, mu *sync.Mutex, text string, imageURL string, imgCache *requestImageEmbeddingCache) {
	start := time.Now()
	classifyResults, err := c.classifyComplexity(ctx, text, imageURL, imgCache)
	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()

	// Record signal extraction metrics for each matched rule
	results.Metrics.Complexity.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0

	logging.Debugf("[Signal Computation] Complexity signal evaluation completed in %v", elapsed)
	if err != nil {
		logging.Errorf("complexity rule evaluation failed: %v", err)
		c.recordComplexityFailure(results, mu)
		return
	}

	bestConfidence := 0.0
	mu.Lock()
	for _, result := range classifyResults {
		matchName := fmt.Sprintf("%s:%s", result.RuleName, result.Difficulty)
		c.recordSignalExtraction(config.SignalTypeComplexity, matchName, latencySeconds)
		c.recordSignalMatch(config.SignalTypeComplexity, matchName)
		metrics.RecordComplexityVerdict(result.RuleName, result.Difficulty, result.SignalSource)
		results.MatchedComplexityRules = append(results.MatchedComplexityRules, matchName)
		// A signal that reports no confidence leaves the key absent, which the
		// decision engine reads as its structural default while marking the
		// pool unscored. Writing a zero here would instead rank the decision
		// last among scored competitors.
		if result.ConfidenceReported {
			results.SignalConfidences["complexity:"+matchName] = result.Confidence
		}
		publishComplexityValues(results.SignalValues, result)
		if result.ConfidenceReported && result.Confidence > bestConfidence {
			bestConfidence = result.Confidence
		}
	}
	results.Metrics.Complexity.Confidence = bestConfidence
	mu.Unlock()
}

// classifyComplexity picks the path that produces the rule results. Both
// remote contracts return the same shape as the local classifier, so
// everything downstream - match names, metrics, published values - is shared
// rather than reimplemented per path.
func (c *Classifier) classifyComplexity(
	ctx context.Context,
	text string,
	imageURL string,
	imgCache *requestImageEmbeddingCache,
) ([]ComplexityRuleResult, error) {
	switch {
	case c.complexityScoreBackend != nil:
		return evaluateComplexityScore(ctx, c.complexityScoreBackend, text, c.complexityRules())
	case c.complexityLabelBackend != nil:
		return evaluateComplexityLabels(ctx, c.complexityLabelBackend, text, c.complexityRules())
	default:
		return c.complexityClassifier.classifyDetailedWithImageCached(text, imageURL, imgCache)
	}
}

// complexityRules reads the rules the remote paths interpret. Only those paths
// need them - the local classifier already holds its own copy - so this is not
// consulted when no backend is configured.
func (c *Classifier) complexityRules() []config.ComplexityRule {
	if c.Config == nil {
		return nil
	}
	return c.Config.ComplexityRules
}
