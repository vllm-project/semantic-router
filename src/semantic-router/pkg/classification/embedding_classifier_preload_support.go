package classification

import (
	"fmt"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type embeddingPreloadResult struct {
	candidate string
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
	return c.ensureCandidateEmbeddings()
}

// preloadCandidateEmbeddings computes embeddings for all unique candidates across all rules.
// Uses concurrent processing for better performance.
func (c *EmbeddingClassifier) preloadCandidateEmbeddings() error {
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
	candidateEmbeddings, successCount, firstError := c.collectCandidateEmbeddingResults(
		c.startCandidateEmbeddingWorkers(candidates, modelType, numWorkers),
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
	c.rebuildRulePrototypeBanks()
	return nil
}

func (c *EmbeddingClassifier) collectUniqueCandidates() []string {
	uniqueCandidates := make(map[string]struct{})
	for _, rule := range c.rules {
		for _, candidate := range rule.Candidates {
			uniqueCandidates[candidate] = struct{}{}
		}
	}

	candidates := make([]string, 0, len(uniqueCandidates))
	for candidate := range uniqueCandidates {
		candidates = append(candidates, candidate)
	}
	return candidates
}

func (c *EmbeddingClassifier) preloadWorkerCount(candidateCount int) int {
	if candidateCount <= 1 {
		return 1
	}
	if strings.EqualFold(c.getBackend(), "candle") {
		return 1
	}
	numWorkers := runtime.NumCPU() * 2
	if numWorkers > candidateCount {
		return candidateCount
	}
	return numWorkers
}

func (c *EmbeddingClassifier) startCandidateEmbeddingWorkers(
	candidates []string,
	modelType string,
	numWorkers int,
) <-chan embeddingPreloadResult {
	resultChan := make(chan embeddingPreloadResult, len(candidates))
	candidateChan := make(chan string, len(candidates))

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
				embedding, err := c.computeEmbedding(candidate, modelType, "preload")
				if err != nil {
					resultChan <- embeddingPreloadResult{candidate: candidate, err: err}
					continue
				}
				resultChan <- embeddingPreloadResult{candidate: candidate, embedding: embedding}
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
) (map[string][]float32, int, error) {
	candidateEmbeddings := make(map[string][]float32)
	var firstError error
	successCount := 0
	for res := range resultChan {
		if res.err != nil {
			if firstError == nil {
				firstError = fmt.Errorf("failed to compute embedding for candidate %q: %w", res.candidate, res.err)
			}
			logging.Warnf("Failed to compute embedding for candidate %q: %v", res.candidate, res.err)
			continue
		}
		candidateEmbeddings[res.candidate] = res.embedding
		successCount++
	}
	return candidateEmbeddings, successCount, firstError
}
