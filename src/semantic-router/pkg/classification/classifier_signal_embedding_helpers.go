package classification

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

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
