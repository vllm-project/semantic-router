package classification

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type embeddingCandidate struct {
	value string
	image bool
}

type embeddingPreloadResult struct {
	candidate embeddingCandidate
	embedding []float32
	err       error
}

// WarmupCandidateEmbeddings eagerly computes candidate embeddings when
// preloading is enabled. Constructors intentionally do not call this because
// model-backed FFI runtimes must be initialized before warmup starts.
func (c *EmbeddingClassifier) WarmupCandidateEmbeddings() error {
	if c == nil {
		return fmt.Errorf("embedding classifier is nil")
	}
	if !c.preloadRequested {
		logging.ComponentDebugEvent("classifier", "embedding_candidates_preload_skipped", map[string]interface{}{
			"reason": "preload_disabled",
		})
		return nil
	}
	return c.ensureCandidateEmbeddings(context.Background())
}

// preloadCandidateEmbeddings computes embeddings for all unique candidates across all rules.
// Uses concurrent processing for better performance.
func (c *EmbeddingClassifier) preloadCandidateEmbeddings(ctx context.Context) error {
	startTime := time.Now()
	candidates := c.collectUniqueCandidates()
	if len(candidates) == 0 {
		logging.ComponentDebugEvent("classifier", "embedding_candidates_preload_skipped", map[string]interface{}{
			"reason": "no_candidates",
		})
		return nil
	}

	modelType := c.getModelType()
	logging.ComponentDebugEvent("classifier", "embedding_candidates_preload_started", map[string]interface{}{
		"candidates":       len(candidates),
		"model_type":       modelType,
		"target_dimension": c.optimizationConfig.TargetDimension,
	})

	numWorkers := c.preloadWorkerCount(len(candidates))
	candidateEmbeddings, imageEmbeddings, successCount, firstError := c.collectCandidateEmbeddingResults(
		c.startCandidateEmbeddingWorkers(ctx, candidates, modelType, numWorkers),
	)

	elapsed := time.Since(startTime)
	logging.ComponentEvent("classifier", "embedding_candidates_preloaded", map[string]interface{}{
		"candidates":       successCount,
		"total_candidates": len(candidates),
		"model_type":       modelType,
		"target_dimension": c.optimizationConfig.TargetDimension,
		"workers":          numWorkers,
		"elapsed_ms":       elapsed.Milliseconds(),
	})

	if firstError != nil {
		return firstError
	}

	c.candidateEmbeddings = candidateEmbeddings
	c.imageCandidateEmbeddings = imageEmbeddings
	c.rebuildRulePrototypeBanks()
	return nil
}

func (c *EmbeddingClassifier) collectUniqueCandidates() []embeddingCandidate {
	unique := make(map[embeddingCandidate]struct{})
	for _, rule := range c.rules {
		for _, values := range [][]string{rule.Candidates, rule.NegativeCandidates} {
			for _, value := range values {
				unique[embeddingCandidate{value: value}] = struct{}{}
			}
		}
		for _, values := range [][]string{rule.ImageCandidates, rule.NegativeImageCandidates} {
			for _, value := range values {
				unique[embeddingCandidate{value: value, image: true}] = struct{}{}
			}
		}
	}
	candidates := make([]embeddingCandidate, 0, len(unique))
	for candidate := range unique {
		candidates = append(candidates, candidate)
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].image != candidates[j].image {
			return !candidates[i].image
		}
		return candidates[i].value < candidates[j].value
	})
	return candidates
}

// Local prepared sessions serialize execution; bound image decode/read work to
// the same single admission slot. Remote text requests use at most four workers.
func (c *EmbeddingClassifier) preloadWorkerCount(candidateCount int) int {
	if candidateCount <= 1 || !strings.EqualFold(c.inferenceBackend(), config.EmbeddingBackendOpenAICompatible) {
		return 1
	}
	if candidateCount < 4 {
		return candidateCount
	}
	return 4
}

func (c *EmbeddingClassifier) startCandidateEmbeddingWorkers(
	ctx context.Context,
	candidates []embeddingCandidate,
	modelType string,
	numWorkers int,
) <-chan embeddingPreloadResult {
	resultChan := make(chan embeddingPreloadResult, len(candidates))
	candidateChan := make(chan embeddingCandidate, len(candidates))

	for _, candidate := range candidates {
		candidateChan <- candidate
	}
	close(candidateChan)

	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for candidate := range candidateChan {
				var vector []float32
				var err error
				if err = ctx.Err(); err != nil {
					resultChan <- embeddingPreloadResult{candidate: candidate, err: err}
					continue
				}
				if candidate.image {
					vector, err = embedding.Image(ctx, c.provider, candidate.value, c.optimizationConfig.TargetDimension)
				} else {
					vector, err = c.computeEmbedding(ctx, candidate.value, modelType, "preload")
				}
				if err == nil {
					err = ctx.Err()
				}
				if err != nil {
					resultChan <- embeddingPreloadResult{candidate: candidate, err: err}
					continue
				}
				resultChan <- embeddingPreloadResult{candidate: candidate, embedding: vector}
			}
		}()
	}

	go func() {
		wg.Wait()
		close(resultChan)
	}()

	return resultChan
}

func (c *EmbeddingClassifier) collectCandidateEmbeddingResults(
	resultChan <-chan embeddingPreloadResult,
) (map[string][]float32, map[string][]float32, int, error) {
	candidateEmbeddings := make(map[string][]float32)
	imageEmbeddings := make(map[string][]float32)
	var firstError error
	successCount := 0
	dimension := 0
	for res := range resultChan {
		if res.err == nil {
			if len(res.embedding) == 0 || (dimension > 0 && len(res.embedding) != dimension) {
				res.err = fmt.Errorf("candidate dimension %d differs from prepared bank dimension %d", len(res.embedding), dimension)
			} else {
				dimension = len(res.embedding)
			}
		}
		if res.err != nil {
			if firstError == nil {
				firstError = fmt.Errorf("failed to compute embedding for candidate %q: %w", res.candidate.value, res.err)
			}
			logging.Warnf("Failed to compute embedding for candidate %q: %v", res.candidate.value, res.err)
			continue
		}
		if res.candidate.image {
			imageEmbeddings[res.candidate.value] = res.embedding
		} else {
			candidateEmbeddings[res.candidate.value] = res.embedding
		}
		successCount++
	}
	return candidateEmbeddings, imageEmbeddings, successCount, firstError
}
