package classification

import (
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func (c *Classifier) evaluateEmbeddingSignal(results *SignalResults, mu *sync.Mutex, text string, imageURL string, imgCache *requestImageEmbeddingCache) {
	start := time.Now()

	// Text-modality evaluation: scores rules whose query_modality is unset
	// or "text". Skipped when the request has no text (image-only content
	// arrays) because ClassifyDetailed rejects an empty query and the error
	// would be misleading - "no text rules to evaluate" is the correct
	// behavior, not a failure.
	var (
		textResult  *EmbeddingClassificationResult
		textErr     error
		textElapsed time.Duration
	)
	if strings.TrimSpace(text) != "" {
		textStart := time.Now()
		textResult, textErr = c.keywordEmbeddingClassifier.ClassifyDetailed(text)
		textElapsed = time.Since(textStart)
	}

	// Image-modality evaluation: only fires when the request carries an
	// image attachment. The classifier's internal rulesByModality cache
	// makes the no-image-rules case a free no-op (returns an empty result
	// without computing the FFI embedding), so this call is safe even when
	// no image rules are configured. The shared imgCache deduplicates the
	// FFI encode against any sibling signal (e.g. complexity image rules)
	// resolving the same image during this request.
	var (
		imageResult  *EmbeddingClassificationResult
		imageErr     error
		imageElapsed time.Duration
	)
	if strings.TrimSpace(imageURL) != "" {
		imageStart := time.Now()
		imageResult, imageErr = c.keywordEmbeddingClassifier.classifyDetailedMultimodalWithCache(config.QueryModalityImage, imageURL, imgCache)
		imageElapsed = time.Since(imageStart)
	}

	elapsed := time.Since(start)

	results.Metrics.Embedding.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	logging.Debugf("[Signal Computation] Embedding signal evaluation completed in %v (text=%v image=%v)",
		elapsed, textElapsed, imageElapsed)

	// Text and image classifications are independent: a failure in one does
	// not skip the other. Pre-PR-2 this function returned early on text
	// error because there was no second classification to attempt. Now there
	// is, and an early return would silently drop a valid image-rule match
	// whenever text classification hit a transient failure.
	if textErr != nil {
		logging.Errorf("text-modality embedding rule evaluation failed: %v", textErr)
	}
	if imageErr != nil {
		logging.Errorf("image-modality embedding rule evaluation failed: %v", imageErr)
	}

	mu.Lock()
	defer mu.Unlock()

	// Track the best confidence across both modalities for the metric.
	// Per-rule extraction-latency observations use modality-specific elapsed
	// times so an image-bearing request that also matched a text rule does
	// not double-count the image FFI cost into the text-rule sample.
	var bestConfidence float64
	if textResult != nil {
		bestConfidence = recordEmbeddingResult(results, textResult, textElapsed, bestConfidence)
	}
	if imageResult != nil {
		bestConfidence = recordEmbeddingResult(results, imageResult, imageElapsed, bestConfidence)
	}
	results.Metrics.Embedding.Confidence = bestConfidence
}

// recordEmbeddingResult merges scores and matches from a single classification
// result into the shared SignalResults. Used by evaluateEmbeddingSignal to
// fold the text-modality and image-modality result sets into one result struct
// without duplicating the bookkeeping logic.
//
// elapsed is the modality-specific time spent producing this detailedResult,
// not the aggregate evaluator time. The caller measures each modality pass
// independently so per-rule extraction-latency samples reflect the cost of
// the rule's own modality - mixing the image FFI cost into a text-rule
// sample (or vice versa) would skew embedding latency dashboards on
// image-bearing requests.
//
// Caller must hold the mu used to guard results.
func recordEmbeddingResult(results *SignalResults, detailedResult *EmbeddingClassificationResult, elapsed time.Duration, bestConfidence float64) float64 {
	for _, score := range detailedResult.Scores {
		if score.Score > bestConfidence {
			bestConfidence = score.Score
		}
		results.SignalValues["embedding:"+score.Name] = score.Score
		results.SignalValues["embedding:"+score.Name+":best"] = score.Best
		results.SignalValues["embedding:"+score.Name+":support"] = score.Support
		results.SignalValues["embedding:"+score.Name+":prototype_count"] = float64(score.PrototypeCount)
	}
	for _, mr := range detailedResult.Matches {
		metrics.RecordSignalExtraction(config.SignalTypeEmbedding, mr.RuleName, elapsed.Seconds())
		metrics.RecordSignalMatch(config.SignalTypeEmbedding, mr.RuleName)
		results.MatchedEmbeddingRules = append(results.MatchedEmbeddingRules, mr.RuleName)
		results.SignalConfidences["embedding:"+mr.RuleName] = mr.Score

		logging.Debugf("[Signal Computation] Embedding match: rule=%q, score=%.4f, method=%s",
			mr.RuleName, mr.Score, mr.Method)
	}
	return bestConfidence
}
