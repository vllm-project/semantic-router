package classification

import (
	"fmt"
	"runtime"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ComplexityClassifier performs complexity-based classification using embedding similarity
// Each rule independently classifies difficulty level using hard/easy candidates
// Results are filtered by composer conditions in the classifier layer
type ComplexityClassifier struct {
	rules []config.ComplexityRule

	// Precomputed embeddings for hard and easy candidates
	hardEmbeddings map[string]map[string][]float32 // ruleName -> candidate -> embedding
	easyEmbeddings map[string]map[string][]float32 // ruleName -> candidate -> embedding

	modelType string // Model type to use for embeddings ("qwen3" or "gemma")
}

// NewComplexityClassifier creates a new ComplexityClassifier with precomputed candidate embeddings
func NewComplexityClassifier(rules []config.ComplexityRule, modelType string) (*ComplexityClassifier, error) {
	if modelType == "" {
		modelType = "qwen3" // Default to qwen3
	}

	c := &ComplexityClassifier{
		rules:          rules,
		hardEmbeddings: make(map[string]map[string][]float32),
		easyEmbeddings: make(map[string]map[string][]float32),
		modelType:      modelType,
	}

	logging.Infof("ComplexityClassifier initialized with model type: %s", c.modelType)

	// Precompute all candidate embeddings at initialization
	if err := c.preloadCandidateEmbeddings(); err != nil {
		logging.Warnf("Failed to preload complexity candidate embeddings: %v", err)
		return nil, err
	}

	return c, nil
}

// preloadCandidateEmbeddings computes embeddings for all hard/easy candidates
// Uses concurrent processing for better performance
func (c *ComplexityClassifier) preloadCandidateEmbeddings() error {
	startTime := time.Now()

	logging.Infof("[Complexity Signal] Preloading embeddings for hard/easy candidates using model: %s with concurrent processing...", c.modelType)

	// Collect all candidates to process
	type candidateTask struct {
		ruleName  string
		candidate string
		isHard    bool
	}

	var tasks []candidateTask
	for _, rule := range c.rules {
		// Initialize maps for this rule
		c.hardEmbeddings[rule.Name] = make(map[string][]float32)
		c.easyEmbeddings[rule.Name] = make(map[string][]float32)

		// Collect hard candidates
		for _, candidate := range rule.Hard.Candidates {
			tasks = append(tasks, candidateTask{
				ruleName:  rule.Name,
				candidate: candidate,
				isHard:    true,
			})
		}

		// Collect easy candidates
		for _, candidate := range rule.Easy.Candidates {
			tasks = append(tasks, candidateTask{
				ruleName:  rule.Name,
				candidate: candidate,
				isHard:    false,
			})
		}
	}

	if len(tasks) == 0 {
		logging.Infof("[Complexity Signal] No candidates to preload")
		return nil
	}

	// Use worker pool for concurrent embedding generation
	numWorkers := runtime.NumCPU() * 2
	if numWorkers > len(tasks) {
		numWorkers = len(tasks)
	}

	type result struct {
		ruleName  string
		candidate string
		embedding []float32
		isHard    bool
		err       error
	}

	resultChan := make(chan result, len(tasks))
	taskChan := make(chan candidateTask, len(tasks))

	// Send all tasks to channel
	for _, task := range tasks {
		taskChan <- task
	}
	close(taskChan)

	// Start workers
	var wg sync.WaitGroup
	var mu sync.Mutex // Protect map writes

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for task := range taskChan {
				output, err := getEmbeddingWithModelType(task.candidate, c.modelType, 0)
				if err != nil {
					resultChan <- result{
						ruleName:  task.ruleName,
						candidate: task.candidate,
						isHard:    task.isHard,
						err:       err,
					}
				} else {
					resultChan <- result{
						ruleName:  task.ruleName,
						candidate: task.candidate,
						embedding: output.Embedding,
						isHard:    task.isHard,
					}
				}
			}
		}(i)
	}

	// Close result channel when all workers are done
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Collect results
	var firstError error
	successCount := 0
	for res := range resultChan {
		if res.err != nil {
			if firstError == nil {
				firstError = fmt.Errorf("failed to compute embedding for %s candidate '%s': %w",
					map[bool]string{true: "hard", false: "easy"}[res.isHard], res.candidate, res.err)
			}
			logging.Warnf("Failed to compute embedding for %s candidate '%s': %v",
				map[bool]string{true: "hard", false: "easy"}[res.isHard], res.candidate, res.err)
		} else {
			mu.Lock()
			if res.isHard {
				c.hardEmbeddings[res.ruleName][res.candidate] = res.embedding
			} else {
				c.easyEmbeddings[res.ruleName][res.candidate] = res.embedding
			}
			mu.Unlock()
			successCount++
		}
	}

	elapsed := time.Since(startTime)
	logging.Infof("[Complexity Signal] Preloaded %d/%d complexity embeddings (hard/easy candidates) using model %s in %v (workers: %d)",
		successCount, len(tasks), c.modelType, elapsed, numWorkers)

	if firstError != nil {
		return firstError
	}

	return nil
}

// Classify evaluates the query against ALL complexity rules independently
// Each rule computes its own difficulty level based on hard/easy candidate similarity
// Returns: all matched rules in format "rulename:difficulty" (e.g., ["code_complexity:hard", "math_complexity:easy"])
// Note: Results will be filtered by composer conditions in the classifier layer (if configured)
func (c *ComplexityClassifier) Classify(query string) ([]string, error) {
	if len(c.rules) == 0 {
		return nil, nil
	}

	// Compute query embedding once
	queryOutput, err := getEmbeddingWithModelType(query, c.modelType, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to compute query embedding: %w", err)
	}
	queryEmbedding := queryOutput.Embedding

	var matchedRules []string

	// Evaluate each rule independently
	for i := range c.rules {
		rule := &c.rules[i]

		// Compute max similarity to hard candidates
		maxHardSim := float32(-1.0)
		for _, hardEmb := range c.hardEmbeddings[rule.Name] {
			sim := cosineSimilarity(queryEmbedding, hardEmb)
			if sim > maxHardSim {
				maxHardSim = sim
			}
		}

		// Compute max similarity to easy candidates
		maxEasySim := float32(-1.0)
		for _, easyEmb := range c.easyEmbeddings[rule.Name] {
			sim := cosineSimilarity(queryEmbedding, easyEmb)
			if sim > maxEasySim {
				maxEasySim = sim
			}
		}

		// Compute difficulty signal
		difficultySignal := maxHardSim - maxEasySim

		// Determine difficulty level
		var difficulty string
		if difficultySignal > rule.Threshold {
			difficulty = "hard"
		} else if difficultySignal < -rule.Threshold {
			difficulty = "easy"
		} else {
			difficulty = "medium"
		}

		logging.Infof("Complexity rule '%s': hard_sim=%.3f, easy_sim=%.3f, signal=%.3f, difficulty=%s",
			rule.Name, maxHardSim, maxEasySim, difficultySignal, difficulty)

		matchedRules = append(matchedRules, fmt.Sprintf("%s:%s", rule.Name, difficulty))
	}

	return matchedRules, nil
}
