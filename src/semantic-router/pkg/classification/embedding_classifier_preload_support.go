package classification

import (
	"fmt"
	"runtime"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type embeddingPreloadResult struct {
	candidate string
	embedding []float32
	err       error
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
				output, err := getEmbeddingWithModelType(candidate, modelType, c.optimizationConfig.TargetDimension)
				if err != nil {
					resultChan <- embeddingPreloadResult{candidate: candidate, err: err}
					continue
				}
				resultChan <- embeddingPreloadResult{candidate: candidate, embedding: output.Embedding}
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
) (int, error) {
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
		c.candidateEmbeddings[res.candidate] = res.embedding
		successCount++
	}
	return successCount, firstError
}
