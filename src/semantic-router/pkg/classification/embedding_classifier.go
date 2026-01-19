package classification

import (
	"fmt"
	"os"
	"sync"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/hnsw"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// getEmbeddingWithModelType is a package-level variable for computing single embeddings.
// It exists so tests can override it.
var getEmbeddingWithModelType = candle_binding.GetEmbeddingWithModelType

// EmbeddingClassifierInitializer initializes KeywordEmbeddingClassifier for embedding based classification
type EmbeddingClassifierInitializer interface {
	Init(qwen3ModelPath string, gemmaModelPath string, useCPU bool) error
}

type ExternalModelBasedEmbeddingInitializer struct{}

func (c *ExternalModelBasedEmbeddingInitializer) Init(qwen3ModelPath string, gemmaModelPath string, useCPU bool) error {
	err := candle_binding.InitEmbeddingModels(qwen3ModelPath, gemmaModelPath, useCPU)
	if err != nil {
		return err
	}
	logging.Infof("Initialized KeywordEmbedding classifier")
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
	candidateToIndex    map[string]int       // candidate text -> HNSW node index
	indexToCandidate    map[int]string       // HNSW node index -> candidate text

	// HNSW index for O(log n) similarity search
	hnswIndex *hnsw.Index

	// Configuration
	optimizationConfig config.HNSWConfig
	preloadEnabled     bool
	useHNSW            bool
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
		candidateToIndex:    make(map[string]int),
		indexToCandidate:    make(map[int]string),
		optimizationConfig:  optConfig,
		preloadEnabled:      optConfig.PreloadEmbeddings,
		useHNSW:             optConfig.UseHNSW,
		modelType:           optConfig.ModelType, // Use configured model type
	}

	logging.Infof("EmbeddingClassifier initialized with model type: %s", c.modelType)

	// If preloading is enabled, compute all candidate embeddings at startup
	if optConfig.PreloadEmbeddings {
		if err := c.preloadCandidateEmbeddings(); err != nil {
			// Log warning but don't fail - fall back to runtime computation
			logging.Warnf("Failed to preload candidate embeddings, falling back to runtime computation: %v", err)
			c.preloadEnabled = false
			c.useHNSW = false
		}
	}

	return c, nil
}

// preloadCandidateEmbeddings computes embeddings for all unique candidates across all rules
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

	logging.Infof("Preloading embeddings for %d unique candidates...", len(uniqueCandidates))

	// Determine model type
	modelType := c.getModelType()

	// Compute embeddings for each candidate
	idx := 0
	for candidate := range uniqueCandidates {
		output, err := getEmbeddingWithModelType(candidate, modelType, c.optimizationConfig.TargetDimension)
		if err != nil {
			return fmt.Errorf("failed to compute embedding for candidate %q: %w", candidate, err)
		}

		c.candidateEmbeddings[candidate] = output.Embedding
		c.candidateToIndex[candidate] = idx
		c.indexToCandidate[idx] = candidate
		idx++
	}

	// Build HNSW index if enabled and we have enough candidates
	hnswThreshold := *c.optimizationConfig.HNSWThreshold // Dereference pointer
	if c.useHNSW && len(uniqueCandidates) >= hnswThreshold {
		c.hnswIndex = hnsw.NewIndex(hnsw.Config{
			M:              c.optimizationConfig.HNSWM,
			EfConstruction: c.optimizationConfig.HNSWEfConstruction,
			EfSearch:       c.optimizationConfig.HNSWEfSearch,
		})

		// Add all candidates to HNSW index
		for candidate, embedding := range c.candidateEmbeddings {
			c.hnswIndex.Add(c.candidateToIndex[candidate], embedding)
		}

		logging.Infof("Built HNSW index with %d nodes (M=%d, efConstruction=%d)",
			c.hnswIndex.Size(), c.optimizationConfig.HNSWM, c.optimizationConfig.HNSWEfConstruction)
	} else {
		c.useHNSW = false // Disable HNSW if not enough candidates
	}

	elapsed := time.Since(startTime)
	logging.Infof("Preloaded %d candidate embeddings in %v (HNSW enabled: %v)",
		len(c.candidateEmbeddings), elapsed, c.useHNSW)

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
	return c.keywordEmbeddingInitializer.Init(c.Config.Qwen3ModelPath, c.Config.GemmaModelPath, c.Config.EmbeddingModels.UseCPU)
}

// ruleResult holds the result of matching a single rule
type ruleResult struct {
	ruleName string
	matched  bool
	score    float32
	err      error
}

// Classify performs Embedding similarity classification on the given text.
// Uses concurrent processing for better performance when multiple rules are present.
func (c *EmbeddingClassifier) Classify(text string) (string, float64, error) {
	// For single rule or very few rules, use sequential processing
	if len(c.rules) <= 1 {
		return c.classifySequential(text)
	}

	// Use concurrent processing for multiple rules
	return c.classifyConcurrent(text)
}

// classifySequential performs sequential rule matching (legacy behavior)
func (c *EmbeddingClassifier) classifySequential(text string) (string, float64, error) {
	logging.Infof("Sequential rule matching for %d rules", len(c.rules))
	var bestScore float32
	var mostMatchedRule string
	for _, rule := range c.rules {
		matched, aggregatedScore, err := c.matches(text, rule)
		if err != nil {
			return "", 0.0, err
		}
		if matched {
			if len(rule.Candidates) > 0 {
				logging.Infof("Embedding similarity classification matched rule %q with candidates: %v, confidence score %v", rule.Name, rule.Candidates, aggregatedScore)
			} else {
				logging.Infof("Embedding similarity classification do not match rule %q with candidates: %v, confidence score %v", rule.Name, rule.Candidates, aggregatedScore)
			}
			if aggregatedScore > bestScore {
				bestScore = aggregatedScore
				mostMatchedRule = rule.Name
			}
		}
	}
	return mostMatchedRule, float64(bestScore), nil
}

// classifyConcurrent performs concurrent rule matching for better performance
func (c *EmbeddingClassifier) classifyConcurrent(text string) (string, float64, error) {
	logging.Infof("Concurrent rule matching for %d rules", len(c.rules))
	results := make(chan ruleResult, len(c.rules))
	var wg sync.WaitGroup

	// Launch goroutine for each rule
	for _, rule := range c.rules {
		wg.Add(1)
		go func(r config.EmbeddingRule) {
			defer wg.Done()
			matched, aggregatedScore, err := c.matches(text, r)
			results <- ruleResult{
				ruleName: r.Name,
				matched:  matched,
				score:    aggregatedScore,
				err:      err,
			}
		}(rule)
	}

	// Close results channel when all goroutines complete
	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results and find best match
	var bestScore float32
	var mostMatchedRule string
	var firstError error

	for result := range results {
		if result.err != nil && firstError == nil {
			firstError = result.err
			continue
		}
		if result.matched {
			// Find the rule to get candidates for logging
			var candidates []string
			for _, rule := range c.rules {
				if rule.Name == result.ruleName {
					candidates = rule.Candidates
					break
				}
			}

			if len(candidates) > 0 {
				logging.Infof("Embedding similarity matched rule %q with candidates: %v, confidence score %v", result.ruleName, candidates, result.score)
			} else {
				logging.Infof("Embedding similarity do not match rule %q with candidates: %v, confidence score %v", result.ruleName, candidates, result.score)
			}

			if result.score > bestScore {
				bestScore = result.score
				mostMatchedRule = result.ruleName
			}
		}
	}

	if firstError != nil {
		return "", 0.0, firstError
	}

	return mostMatchedRule, float64(bestScore), nil
}

// matches checks if the text matches the given keyword rule.
// Uses preloaded embeddings if available, otherwise falls back to runtime computation.
func (c *EmbeddingClassifier) matches(text string, rule config.EmbeddingRule) (bool, float32, error) {
	// Validate input
	if text == "" {
		return false, 0.0, fmt.Errorf("embedding similarity classification: query must be provided")
	}
	if len(rule.Candidates) == 0 {
		return false, 0.0, fmt.Errorf("embedding similarity classification: candidates must be provided")
	}

	logging.Infof("Embedding similarity classification using optimized path")
	return c.matchesOptimized(text, rule)
}

// matchesOptimized uses preloaded embeddings for fast similarity matching
func (c *EmbeddingClassifier) matchesOptimized(text string, rule config.EmbeddingRule) (bool, float32, error) {
	// Compute query embedding only
	modelType := c.getModelType()

	// Log search configuration
	logging.Infof("Embedding Search Config - Rule: %q, Model: %s, Threshold: %.3f, Candidates: %d, Dimension: %d",
		rule.Name, modelType, rule.SimilarityThreshold, len(rule.Candidates), c.optimizationConfig.TargetDimension)

	queryOutput, err := getEmbeddingWithModelType(text, modelType, c.optimizationConfig.TargetDimension)
	if err != nil {
		return false, 0.0, fmt.Errorf("failed to compute query embedding: %w", err)
	}
	queryEmbedding := queryOutput.Embedding

	// Calculate similarities against preloaded candidates
	similarities := make([]float32, 0, len(rule.Candidates))

	hnswThreshold := *c.optimizationConfig.HNSWThreshold // Dereference pointer
	if c.useHNSW && c.hnswIndex != nil && len(rule.Candidates) >= hnswThreshold {
		// Use HNSW for O(log n) search
		// Search for all candidates to get complete similarity list
		results := c.hnswIndex.Search(queryEmbedding, len(c.candidateEmbeddings))

		// Filter results to only include candidates from this rule
		candidateSet := make(map[string]bool)
		for _, candidate := range rule.Candidates {
			candidateSet[candidate] = true
		}

		for _, result := range results {
			candidate := c.indexToCandidate[result.ID]
			if candidateSet[candidate] {
				similarities = append(similarities, result.Similarity)
			}
		}

		logging.Infof("Computed %d similarities for rule %q (using preloaded embeddings)",
			len(similarities), rule.Name)
	} else {
		// Use brute-force for small candidate sets
		logging.Infof("Using brute-force search for %d candidates (below HNSW threshold: %d)",
			len(rule.Candidates), hnswThreshold)

		for i, candidate := range rule.Candidates {
			embedding, ok := c.candidateEmbeddings[candidate]
			if !ok {
				// Candidate not preloaded, compute on the fly
				logging.Infof("Computing embedding on-the-fly for candidate: %q (model: %s)", candidate, modelType)
				output, err := getEmbeddingWithModelType(candidate, modelType, c.optimizationConfig.TargetDimension)
				if err != nil {
					return false, 0.0, fmt.Errorf("failed to compute embedding for candidate %q: %w", candidate, err)
				}
				embedding = output.Embedding
				c.candidateEmbeddings[candidate] = embedding // Cache for future use
			}

			sim := cosineSimilarity(queryEmbedding, embedding)
			similarities = append(similarities, sim)

			// Debug: log first 3 similarities
			logging.Infof("[Brute-force path] candidate[%d]=%q, similarity=%.4f", i, candidate, sim)
		}
	}

	// Aggregate scores based on method
	return c.aggregateScores(similarities, rule)
}

// aggregateScores applies the aggregation method to compute the final score
func (c *EmbeddingClassifier) aggregateScores(similarities []float32, rule config.EmbeddingRule) (bool, float32, error) {
	if len(similarities) == 0 {
		return false, 0.0, nil
	}

	switch rule.AggregationMethodConfiged {
	case config.AggregationMethodMean:
		var aggregatedScore float32
		for _, sim := range similarities {
			aggregatedScore += sim
		}
		aggregatedScore /= float32(len(similarities))
		matched := aggregatedScore >= rule.SimilarityThreshold
		logging.Infof("Aggregation (Mean): score=%.4f, threshold=%.3f, matched=%v",
			aggregatedScore, rule.SimilarityThreshold, matched)
		return matched, aggregatedScore, nil

	case config.AggregationMethodMax:
		var aggregatedScore float32
		for _, sim := range similarities {
			if sim > aggregatedScore {
				aggregatedScore = sim
			}
		}
		matched := aggregatedScore >= rule.SimilarityThreshold
		logging.Infof("Aggregation (Max): score=%.4f, threshold=%.3f, matched=%v",
			aggregatedScore, rule.SimilarityThreshold, matched)
		return matched, aggregatedScore, nil

	case config.AggregationMethodAny:
		for _, sim := range similarities {
			if sim >= rule.SimilarityThreshold {
				logging.Infof("Aggregation (Any): found match with score=%.4f >= threshold=%.3f",
					sim, rule.SimilarityThreshold)
				return true, rule.SimilarityThreshold, nil
			}
		}
		logging.Infof("Aggregation (Any): no match found (threshold=%.3f)", rule.SimilarityThreshold)
		return false, 0.0, nil

	default:
		return false, 0.0, fmt.Errorf("unsupported aggregation method: %q", rule.AggregationMethodConfiged)
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
func (c *EmbeddingClassifier) GetPreloadStats() (int, bool, int) {
	candidateCount := len(c.candidateEmbeddings)
	hnswEnabled := c.useHNSW
	hnswSize := 0
	if c.hnswIndex != nil {
		hnswSize = c.hnswIndex.Size()
	}
	return candidateCount, hnswEnabled, hnswSize
}
