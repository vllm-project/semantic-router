package classification

import (
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func (c *Classifier) evaluateEmbeddingSignal(results *SignalResults, mu *sync.Mutex, text string, imageURL string, imgCache *requestImageEmbeddingCache, ocrCache *requestImageOCRCache) {
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
		// Determine whether any matched image rules opt into a stage-2 text
		// verification. If none do, fall back to the old behaviour and record
		// the image result wholesale.
		needsStage2 := false
		for _, mr := range imageResult.Matches {
			for _, r := range c.keywordEmbeddingClassifier.rules {
				if r.Name == mr.RuleName && r.EffectiveQueryModality() == config.QueryModalityImage && r.Stage2TextCheck {
					needsStage2 = true
					break
				}
			}
			if needsStage2 {
				break
			}
		}

		if !needsStage2 {
			bestConfidence = recordEmbeddingResult(results, imageResult, imageElapsed, bestConfidence)
		} else {
			// Perform OCR once for the image (cached) and run a text-modality
			// classification over the OCR'd text. If OCR or text-classification
			// fails or yields no match, conservatively drop the image-only
			// matches that requested a stage-2 check.
			ocrText, ocrErr := c.keywordEmbeddingClassifier.ExtractImageOCR(imageURL, ocrCache)
			if ocrErr != nil {
				logging.Errorf("image OCR failed: %v", ocrErr)
				// On OCR failure, record only the image matches that did not opt
				// into stage-2.
				filtered := filterImageResultByStage2(imageResult, c.keywordEmbeddingClassifier, false)
				bestConfidence = recordEmbeddingResult(results, filtered, imageElapsed, bestConfidence)
			} else if strings.TrimSpace(ocrText) == "" {
				// No readable text found: drop stage-2-only matches.
				filtered := filterImageResultByStage2(imageResult, c.keywordEmbeddingClassifier, false)
				bestConfidence = recordEmbeddingResult(results, filtered, imageElapsed, bestConfidence)
			} else {
				textStart := time.Now()
				textResultFromOCR, textErrFromOCR := c.keywordEmbeddingClassifier.ClassifyDetailed(ocrText)
				textElapsedFromOCR := time.Since(textStart)
				if textErrFromOCR != nil {
					logging.Errorf("text-modality embedding rule evaluation on OCR'd image failed: %v", textErrFromOCR)
					// Conservative: drop stage-2-only matches.
					filtered := filterImageResultByStage2(imageResult, c.keywordEmbeddingClassifier, false)
					bestConfidence = recordEmbeddingResult(results, filtered, imageElapsed, bestConfidence)
				} else {
					// If the OCR'd text produced at least one text-modality match, we
					// consider stage-2 satisfied and keep stage-2 matches; otherwise
					// drop them.
					keepStage2 := len(textResultFromOCR.Matches) > 0
					filtered := filterImageResultByStage2(imageResult, c.keywordEmbeddingClassifier, keepStage2)
					bestConfidence = recordEmbeddingResult(results, filtered, imageElapsed, bestConfidence)
					// Also merge text-result scores into the metrics so the OCR
					// classification's latency is observable separately.
					if len(textResultFromOCR.Scores) > 0 {
						// Record per-rule latencies/metrics for the OCR text pass.
						for _, mr := range textResultFromOCR.Matches {
							metrics.RecordSignalExtraction(config.SignalTypeEmbedding, mr.RuleName, textElapsedFromOCR.Seconds())
							metrics.RecordSignalMatch(config.SignalTypeEmbedding, mr.RuleName)
							results.SignalConfidences["embedding:"+mr.RuleName] = mr.Score
							results.MatchedEmbeddingRules = append(results.MatchedEmbeddingRules, mr.RuleName)
						}
					}
				}
			}
		}
	}
	results.Metrics.Embedding.Confidence = bestConfidence
}

// filterImageResultByStage2 returns a copy of detailedResult retaining only
// the scores and matches that should be kept given keepStage2. If
// keepStage2==true, all matches are retained. If false, any rule that had
// Stage2TextCheck==true and declared QueryModality=image is filtered out.
func filterImageResultByStage2(detailedResult *EmbeddingClassificationResult, ec *EmbeddingClassifier, keepStage2 bool) *EmbeddingClassificationResult {
	if detailedResult == nil {
		return &EmbeddingClassificationResult{}
	}
	if keepStage2 {
		return detailedResult
	}
	// Build a set of rule names to retain.
	retain := make(map[string]bool)
	for _, score := range detailedResult.Scores {
		retain[score.Name] = true
	}
	for _, r := range ec.rules {
		if r.EffectiveQueryModality() == config.QueryModalityImage && r.Stage2TextCheck {
			// Drop stage-2 rules.
			delete(retain, r.Name)
		}
	}

	filteredScores := make([]EmbeddingRuleScore, 0, len(detailedResult.Scores))
	for _, s := range detailedResult.Scores {
		if retain[s.Name] {
			filteredScores = append(filteredScores, s)
		}
	}
	filteredMatches := make([]MatchedRule, 0, len(detailedResult.Matches))
	for _, m := range detailedResult.Matches {
		if retain[m.RuleName] {
			filteredMatches = append(filteredMatches, m)
		}
	}
	return &EmbeddingClassificationResult{Scores: filteredScores, Matches: filteredMatches}
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
