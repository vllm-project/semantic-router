package classification

import (
	"fmt"
	"os"
	"runtime"
	"sync"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// getEmbeddingWithModelType is a package-level variable for computing single embeddings.
// It exists so tests can override it.
var getEmbeddingWithModelType = candle_binding.GetEmbeddingWithModelType

// EmbeddingClassifierInitializer initializes KeywordEmbeddingClassifier for embedding based classification
type EmbeddingClassifierInitializer interface {
	Init(qwen3ModelPath string, gemmaModelPath string, mmBertModelPath string, useCPU bool) error
}

type ExternalModelBasedEmbeddingInitializer struct{}

func (c *ExternalModelBasedEmbeddingInitializer) Init(qwen3ModelPath string, gemmaModelPath string, mmBertModelPath string, useCPU bool) error {
	// Resolve model paths using registry (supports aliases like "qwen3", "gemma", "mmbert")
	qwen3ModelPath = config.ResolveModelPath(qwen3ModelPath)
	gemmaModelPath = config.ResolveModelPath(gemmaModelPath)
	mmBertModelPath = config.ResolveModelPath(mmBertModelPath)

	err := candle_binding.InitEmbeddingModels(qwen3ModelPath, gemmaModelPath, mmBertModelPath, useCPU)
	if err != nil {
		return err
	}

	if mmBertModelPath != "" {
		logging.Infof("Initialized KeywordEmbedding classifier with mmBERT 2D Matryoshka support")
	} else {
		logging.Infof("Initialized KeywordEmbedding classifier")
	}
	return nil
}

// createEmbeddingInitializer creates the appropriate keyword embedding initializer based on configuration
func createEmbeddingInitializer() EmbeddingClassifierInitializer {
	return &ExternalModelBasedEmbeddingInitializer{}
}

// EmbeddingClassifier performs embedding-based similarity classification.
// When preloading is enabled, candidate embeddings are computed once at initialization
// and reused for all classification requests, significantly improving performance.
type EmbeddingClassifier struct {
	rules []config.EmbeddingRule

	// Optimization: preloaded candidate embeddings
	candidateEmbeddings map[string][]float32 // candidate text -> embedding vector

	// Configuration
	optimizationConfig config.HNSWConfig
	preloadEnabled     bool
	modelType          string // Model type to use for embeddings ("qwen3" or "gemma")
}

// NewEmbeddingClassifier creates a new EmbeddingClassifier.
// If optimization config has PreloadEmbeddings enabled, candidate embeddings
// will be precomputed at initialization time for better runtime performance.
func NewEmbeddingClassifier(cfgRules []config.EmbeddingRule, optConfig config.HNSWConfig) (*EmbeddingClassifier, error) {
	// Apply defaults
	optConfig = optConfig.WithDefaults()

	c := &EmbeddingClassifier{
		rules:               cfgRules,
		candidateEmbeddings: make(map[string][]float32),
		optimizationConfig:  optConfig,
		preloadEnabled:      optConfig.PreloadEmbeddings,
		modelType:           optConfig.ModelType, // Use configured model type
	}

	logging.Infof("EmbeddingClassifier initialized with model type: %s", c.modelType)

	// If preloading is enabled, compute all candidate embeddings at startup
	if optConfig.PreloadEmbeddings {
		if err := c.preloadCandidateEmbeddings(); err != nil {
			// Log warning but don't fail - fall back to runtime computation
			logging.Warnf("Failed to preload candidate embeddings, falling back to runtime computation: %v", err)
			c.preloadEnabled = false
		}
	}

	return c, nil
}

// preloadCandidateEmbeddings computes embeddings for all unique candidates across all rules
// Uses concurrent processing for better performance
func (c *EmbeddingClassifier) preloadCandidateEmbeddings() error {
	startTime := time.Now()

	// Collect all unique candidates
	uniqueCandidates := make(map[string]bool)
	for _, rule := range c.rules {
		for _, candidate := range rule.Candidates {
			uniqueCandidates[candidate] = true
		}
	}

	if len(uniqueCandidates) == 0 {
		logging.Infof("No candidates to preload")
		return nil
	}

	// Determine model type
	modelType := c.getModelType()

	logging.Infof("[Embedding Signal] Preloading embeddings for %d unique candidates using model: %s (dimension: %d) with concurrent processing...",
		len(uniqueCandidates), modelType, c.optimizationConfig.TargetDimension)

	// Convert map to slice for concurrent processing
	candidates := make([]string, 0, len(uniqueCandidates))
	for candidate := range uniqueCandidates {
		candidates = append(candidates, candidate)
	}

	// Use worker pool for concurrent embedding generation
	numWorkers := runtime.NumCPU() * 2
	if numWorkers > len(candidates) {
		numWorkers = len(candidates)
	}

	type result struct {
		candidate string
		embedding []float32
		err       error
	}

	resultChan := make(chan result, len(candidates))
	candidateChan := make(chan string, len(candidates))

	// Send all candidates to channel
	for _, candidate := range candidates {
		candidateChan <- candidate
	}
	close(candidateChan)

	// Start workers
	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for candidate := range candidateChan {
				output, err := getEmbeddingWithModelType(candidate, modelType, c.optimizationConfig.TargetDimension)
				if err != nil {
					resultChan <- result{candidate: candidate, err: err}
				} else {
					resultChan <- result{candidate: candidate, embedding: output.Embedding}
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
				firstError = fmt.Errorf("failed to compute embedding for candidate %q: %w", res.candidate, res.err)
			}
			logging.Warnf("Failed to compute embedding for candidate %q: %v", res.candidate, res.err)
		} else {
			c.candidateEmbeddings[res.candidate] = res.embedding
			successCount++
		}
	}

	elapsed := time.Since(startTime)
	logging.Infof("[Embedding Signal] Preloaded %d/%d candidate embeddings using model %s in %v (workers: %d)",
		successCount, len(candidates), modelType, elapsed, numWorkers)

	if firstError != nil {
		return firstError
	}

	return nil
}

// getModelType returns the model type to use for embeddings
func (c *EmbeddingClassifier) getModelType() string {
	// Check for test override via environment variable
	if model := os.Getenv("EMBEDDING_MODEL_OVERRIDE"); model != "" {
		logging.Infof("Embedding model override from env: %s", model)
		return model
	}
	// Use the configured model type from config
	// This ensures consistency between preload and runtime
	return c.modelType
}

// IsKeywordEmbeddingClassifierEnabled checks if Keyword embedding classification rules are properly configured
func (c *Classifier) IsKeywordEmbeddingClassifierEnabled() bool {
	return len(c.Config.EmbeddingRules) > 0
}

// initializeKeywordEmbeddingClassifier initializes the KeywordEmbedding classification model
func (c *Classifier) initializeKeywordEmbeddingClassifier() error {
	if !c.IsKeywordEmbeddingClassifierEnabled() || c.keywordEmbeddingInitializer == nil {
		return fmt.Errorf("keyword embedding similarity match is not properly configured")
	}

	// Initialize with all three model paths (qwen3, gemma, mmbert)
	// The Init method will handle path resolution and choose the appropriate FFI function
	return c.keywordEmbeddingInitializer.Init(
		c.Config.Qwen3ModelPath,
		c.Config.GemmaModelPath,
		c.Config.EmbeddingModels.MmBertModelPath,
		c.Config.EmbeddingModels.UseCPU,
	)
}

// Classify performs Embedding similarity classification on the given text.
// New implementation: computes query embedding once, searches all candidates once,
// then distributes results to rules based on topK matches.
func (c *EmbeddingClassifier) Classify(text string) (string, float64, error) {
	if len(c.rules) == 0 {
		return "", 0.0, nil
	}

	// Validate input
	if text == "" {
		return "", 0.0, fmt.Errorf("embedding similarity classification: query must be provided")
	}

	startTime := time.Now()

	// Step 1: Compute query embedding once
	modelType := c.getModelType()
	queryOutput, err := getEmbeddingWithModelType(text, modelType, c.optimizationConfig.TargetDimension)
	if err != nil {
		return "", 0.0, fmt.Errorf("failed to compute query embedding: %w", err)
	}
	queryEmbedding := queryOutput.Embedding

	logging.Infof("Computed query embedding (model: %s, dimension: %d)", modelType, len(queryEmbedding))

	// Step 2: Search all candidates once and get similarities
	candidateSimilarities, err := c.searchAllCandidates(queryEmbedding)
	if err != nil {
		return "", 0.0, err
	}

	logging.Infof("Computed %d candidate similarities in %v", len(candidateSimilarities), time.Since(startTime))

	// Step 3: Aggregate scores per rule and find best match
	bestRule, bestScore, err := c.findBestRule(candidateSimilarities)
	if err != nil {
		return "", 0.0, err
	}

	elapsed := time.Since(startTime)
	logging.Infof("Classification completed in %v: rule=%q, score=%.4f", elapsed, bestRule, bestScore)

	return bestRule, float64(bestScore), nil
}

// searchAllCandidates computes similarities for all candidates in one pass
// Always uses brute-force to ensure we get ALL candidate similarities
func (c *EmbeddingClassifier) searchAllCandidates(queryEmbedding []float32) (map[string]float32, error) {
	candidateSimilarities := make(map[string]float32)
	totalCandidates := len(c.candidateEmbeddings)

	// For embedding classification, we MUST compute similarities for ALL candidates
	// to correctly aggregate scores per rule and find the best match.
	// HNSW is an approximate algorithm designed for topK search, not exhaustive search.
	// Even with large ef values, HNSW may miss some candidates due to graph connectivity.
	//
	// Brute-force is the right choice here because:
	// 1. We need complete results (all candidates), not approximate topK
	// 2. Candidate sets are typically small (50-200), making brute-force very fast
	// 3. Embeddings are pre-loaded in memory, so it's just dot products (microseconds each)
	// 4. Simpler and more reliable than tuning HNSW parameters

	logging.Infof("Computing similarities for all %d candidates (brute-force)", totalCandidates)

	for candidate, embedding := range c.candidateEmbeddings {
		sim := cosineSimilarity(queryEmbedding, embedding)
		candidateSimilarities[candidate] = sim
		logging.Debugf("[Brute-force] candidate=%q, similarity=%.4f", candidate, sim)
	}

	return candidateSimilarities, nil
}

// ruleScore holds the aggregated score for a rule
type ruleScore struct {
	ruleName string
	score    float32
	matched  bool // whether it meets the hard threshold
}

// findBestRule aggregates candidate similarities per rule and finds the best match
func (c *EmbeddingClassifier) findBestRule(candidateSimilarities map[string]float32) (string, float32, error) {
	ruleScores := make([]ruleScore, 0, len(c.rules))

	// Aggregate scores for each rule
	for _, rule := range c.rules {
		if len(rule.Candidates) == 0 {
			continue
		}

		// Collect similarities for this rule's candidates
		similarities := make([]float32, 0, len(rule.Candidates))
		for _, candidate := range rule.Candidates {
			if sim, ok := candidateSimilarities[candidate]; ok {
				similarities = append(similarities, sim)
			}
		}

		if len(similarities) == 0 {
			continue
		}

		// Aggregate based on method
		aggregatedScore := c.aggregateScoresForRule(similarities, rule.AggregationMethodConfiged)
		matched := aggregatedScore >= rule.SimilarityThreshold

		logging.Infof("Rule %q: aggregated_score=%.4f, threshold=%.3f, matched=%v (method=%s, candidates=%d)",
			rule.Name, aggregatedScore, rule.SimilarityThreshold, matched,
			rule.AggregationMethodConfiged, len(similarities))

		ruleScores = append(ruleScores, ruleScore{
			ruleName: rule.Name,
			score:    aggregatedScore,
			matched:  matched,
		})
	}

	if len(ruleScores) == 0 {
		return "", 0.0, nil
	}

	// Find best match using hard threshold first
	var bestHardMatch *ruleScore
	for i := range ruleScores {
		if ruleScores[i].matched {
			if bestHardMatch == nil || ruleScores[i].score > bestHardMatch.score {
				bestHardMatch = &ruleScores[i]
			}
		}
	}

	if bestHardMatch != nil {
		logging.Infof("Hard match found: rule=%q, score=%.4f", bestHardMatch.ruleName, bestHardMatch.score)
		return bestHardMatch.ruleName, bestHardMatch.score, nil
	}

	// No hard match - check if soft matching is enabled
	if c.optimizationConfig.EnableSoftMatching == nil || !*c.optimizationConfig.EnableSoftMatching {
		logging.Infof("No hard match found and soft matching is disabled")
		return "", 0.0, nil
	}

	// Find best soft match
	var bestSoftMatch *ruleScore
	for i := range ruleScores {
		if ruleScores[i].score >= c.optimizationConfig.MinScoreThreshold {
			if bestSoftMatch == nil || ruleScores[i].score > bestSoftMatch.score {
				bestSoftMatch = &ruleScores[i]
			}
		}
	}

	if bestSoftMatch != nil {
		logging.Infof("Soft match found: rule=%q, score=%.4f (min_threshold=%.3f)",
			bestSoftMatch.ruleName, bestSoftMatch.score, c.optimizationConfig.MinScoreThreshold)
		return bestSoftMatch.ruleName, bestSoftMatch.score, nil
	}

	logging.Infof("No match found (best score below min_threshold=%.3f)", c.optimizationConfig.MinScoreThreshold)
	return "", 0.0, nil
}

// aggregateScoresForRule applies the aggregation method to compute the final score
func (c *EmbeddingClassifier) aggregateScoresForRule(similarities []float32, method config.AggregationMethod) float32 {
	if len(similarities) == 0 {
		return 0.0
	}

	switch method {
	case config.AggregationMethodMean:
		var sum float32
		for _, sim := range similarities {
			sum += sim
		}
		return sum / float32(len(similarities))

	case config.AggregationMethodMax:
		var max float32
		for _, sim := range similarities {
			if sim > max {
				max = sim
			}
		}
		return max

	case config.AggregationMethodAny:
		// For "any" method, return the max similarity
		// The threshold check will be done by the caller
		var max float32
		for _, sim := range similarities {
			if sim > max {
				max = sim
			}
		}
		return max

	default:
		logging.Warnf("Unsupported aggregation method: %q, using max", method)
		var max float32
		for _, sim := range similarities {
			if sim > max {
				max = sim
			}
		}
		return max
	}
}

// cosineSimilarity computes cosine similarity between two vectors.
// Assumes vectors are normalized (which they should be from BERT-style models).
func cosineSimilarity(a, b []float32) float32 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}

	minLen := len(a)
	if len(b) < minLen {
		minLen = len(b)
	}

	var dotProduct float32
	for i := 0; i < minLen; i++ {
		dotProduct += a[i] * b[i]
	}

	return dotProduct
}

// GetPreloadStats returns statistics about preloaded embeddings
func (c *EmbeddingClassifier) GetPreloadStats() int {
	return len(c.candidateEmbeddings)
}
