package classification

import (
	"context"
	"math"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type embeddingSignalInput struct{ Text, Image, Audio string }

type modalityEvaluation struct {
	modality config.QueryModality
	result   *EmbeddingClassificationResult
	err      error
	elapsed  time.Duration
}

func (c *Classifier) evaluateEmbeddingSignal(ctx context.Context, results *SignalResults, mu *sync.Mutex, input embeddingSignalInput, mediaCache *requestMediaEmbeddingCache) {
	start := time.Now()
	evaluations := make([]modalityEvaluation, 0, 3)
	for _, query := range []struct {
		modality config.QueryModality
		payload  string
	}{
		{config.QueryModalityText, input.Text}, {config.QueryModalityImage, input.Image}, {config.QueryModalityAudio, input.Audio},
	} {
		if strings.TrimSpace(query.payload) == "" {
			continue
		}
		evaluation := modalityEvaluation{modality: query.modality}
		modalityStart := time.Now()
		if query.modality == config.QueryModalityText {
			evaluation.result, evaluation.err = c.keywordEmbeddingClassifier.ClassifyDetailedWithContext(ctx, query.payload)
		} else {
			evaluation.result, evaluation.err = c.keywordEmbeddingClassifier.classifyDetailedMultimodalWithCache(ctx, query.modality, query.payload, mediaCache)
		}
		evaluation.elapsed = time.Since(modalityStart)
		if evaluation.err != nil {
			logging.Errorf("%s embedding rule evaluation failed: %v", query.modality, evaluation.err)
			c.recordEmbeddingSignalError(results, mu, query.modality)
		}
		evaluations = append(evaluations, evaluation)
	}
	mu.Lock()
	results.Metrics.Embedding.ExecutionTimeMs = float64(time.Since(start).Microseconds()) / 1000
	// A request cancellation invalidates every result. Ordinary modality errors
	// remain independent, so a valid attachment can still select its route.
	if ctx.Err() != nil {
		mu.Unlock()
		for _, evaluation := range evaluations {
			c.recordEmbeddingSignalError(results, mu, evaluation.modality)
		}
		return
	}
	defer mu.Unlock()
	var confidence float64
	for _, evaluation := range evaluations {
		if evaluation.err == nil && evaluation.result != nil {
			confidence = c.recordEmbeddingResult(results, evaluation.result, evaluation.elapsed, confidence)
		}
	}
	results.Metrics.Embedding.Confidence = confidence
}

func (c *Classifier) recordEmbeddingSignalError(results *SignalResults, mu *sync.Mutex, modality config.QueryModality) {
	rules := c.keywordEmbeddingClassifier.rulesByModality[modality]
	names := make([]string, 0, len(rules))
	for _, rule := range rules {
		names = append(names, rule.Name)
	}
	recordSignalRuleErrors(results, mu, config.SignalTypeEmbedding, names, embeddingEvaluationFailedCode)
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
func (c *Classifier) recordEmbeddingResult(results *SignalResults, detailedResult *EmbeddingClassificationResult, elapsed time.Duration, bestConfidence float64) float64 {
	confidences := make(map[string]float64, len(detailedResult.Scores))
	for _, score := range detailedResult.Scores {
		confidence := embeddingSignalConfidence(score.Score, score.Contrastive)
		confidences[score.Name] = confidence
		if confidence > bestConfidence {
			bestConfidence = confidence
		}
		results.SignalValues["embedding:"+score.Name+":positive"] = score.PositiveScore
		results.SignalValues["embedding:"+score.Name+":negative"] = score.NegativeScore
		results.SignalValues["embedding:"+score.Name] = score.Score
		results.SignalValues["embedding:"+score.Name+":best"] = score.Best
		results.SignalValues["embedding:"+score.Name+":support"] = score.Support
		results.SignalValues["embedding:"+score.Name+":prototype_count"] = float64(score.PrototypeCount)
	}
	for _, mr := range detailedResult.Matches {
		c.recordSignalExtraction(config.SignalTypeEmbedding, mr.RuleName, elapsed.Seconds())
		c.recordSignalMatch(config.SignalTypeEmbedding, mr.RuleName)
		results.MatchedEmbeddingRules = append(results.MatchedEmbeddingRules, mr.RuleName)
		confidence, ok := confidences[mr.RuleName]
		if !ok {
			confidence = embeddingSignalConfidence(mr.Score, false)
		}
		results.SignalConfidences["embedding:"+mr.RuleName] = confidence

		logging.Debugf("[Signal Computation] Embedding match: rule=%q, score=%.4f, method=%s",
			mr.RuleName, mr.Score, mr.Method)
	}
	return bestConfidence
}

// Confidence is bounded evidence strength, never a calibrated probability.
// Raw cosine margins remain in SignalValues and drive threshold matching.
func embeddingSignalConfidence(score float64, contrastive bool) float64 {
	if contrastive {
		score = (score + 2) / 4
	}
	return math.Max(0, math.Min(1, score))
}
