package classification

import (
	"fmt"
	"os"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/hnsw"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// calculateSimilarityBatch is a package-level variable that points to the
// actual implementation in the candle_binding package. It exists so tests can
// override it.
var calculateSimilarityBatch = candle_binding.CalculateSimilarityBatch

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
	}

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

// NewEmbeddingClassifierLegacy creates a new EmbeddingClassifier without optimizations.
// This maintains backward compatibility with existing code.
func NewEmbeddingClassifierLegacy(cfgRules []config.EmbeddingRule) (*EmbeddingClassifier, error) {
	return &EmbeddingClassifier{
		rules:               cfgRules,
		candidateEmbeddings: make(map[string][]float32),
		candidateToIndex:    make(map[string]int),
		indexToCandidate:    make(map[int]string),
		preloadEnabled:      false,
		useHNSW:             false,
	}, nil
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
	if c.useHNSW && len(uniqueCandidates) >= c.optimizationConfig.HNSWThreshold {
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
	if testModel := os.Getenv("EMBEDDING_MODEL_OVERRIDE"); testModel != "" {
		logging.Infof("Embedding model override from env: %s", testModel)
		return testModel
	}
	// For preloading, we need a specific model type (not "auto")
	// Default to qwen3 for consistent embeddings
	return "qwen3"
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

// Classify performs keyword-based embedding similarity classification on the given text.
func (c *EmbeddingClassifier) Classify(text string) (string, float64, error) {
	var bestScore float32
	var mostMatchedRule string
	for _, rule := range c.rules {
		matched, aggregatedScore, err := c.matches(text, rule)
		if err != nil {
			return "", 0.0, err
		}
		if matched {
			if len(rule.Candidates) > 0 {
				logging.Infof("Keyword-based embedding similarity classification matched rule %q with candidates: %v, confidence score %v", rule.Name, rule.Candidates, aggregatedScore)
			} else {
				logging.Infof("Keyword-based embedding similarity classification do not match rule %q with candidates: %v, confidence score %v", rule.Name, rule.Candidates, aggregatedScore)
			}
			if aggregatedScore > bestScore {
				bestScore = aggregatedScore
				mostMatchedRule = rule.Name
			}
		}
	}
	return mostMatchedRule, float64(bestScore), nil
}

// matches checks if the text matches the given keyword rule.
// Uses preloaded embeddings if available, otherwise falls back to runtime computation.
func (c *EmbeddingClassifier) matches(text string, rule config.EmbeddingRule) (bool, float32, error) {
	// Validate input
	if text == "" {
		return false, 0.0, fmt.Errorf("keyword-based embedding similarity classification: query must be provided")
	}
	if len(rule.Candidates) == 0 {
		return false, 0.0, fmt.Errorf("keyword-based embedding similarity classification: candidates must be provided")
	}

	// Use optimized path if preloading is enabled
	if c.preloadEnabled {
		return c.matchesOptimized(text, rule)
	}

	// Fall back to legacy implementation
	return c.matchesLegacy(text, rule)
}

// matchesOptimized uses preloaded embeddings for fast similarity matching
func (c *EmbeddingClassifier) matchesOptimized(text string, rule config.EmbeddingRule) (bool, float32, error) {
	// Compute query embedding only
	modelType := c.getModelType()
	queryOutput, err := getEmbeddingWithModelType(text, modelType, c.optimizationConfig.TargetDimension)
	if err != nil {
		return false, 0.0, fmt.Errorf("failed to compute query embedding: %w", err)
	}
	queryEmbedding := queryOutput.Embedding

	// Calculate similarities against preloaded candidates
	similarities := make([]float32, 0, len(rule.Candidates))

	if c.useHNSW && c.hnswIndex != nil && len(rule.Candidates) >= c.optimizationConfig.HNSWThreshold {
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
	} else {
		// Use brute-force for small candidate sets
		for _, candidate := range rule.Candidates {
			embedding, ok := c.candidateEmbeddings[candidate]
			if !ok {
				// Candidate not preloaded, compute on the fly
				output, err := getEmbeddingWithModelType(candidate, modelType, c.optimizationConfig.TargetDimension)
				if err != nil {
					return false, 0.0, fmt.Errorf("failed to compute embedding for candidate %q: %w", candidate, err)
				}
				embedding = output.Embedding
				c.candidateEmbeddings[candidate] = embedding // Cache for future use
			}

			sim := cosineSimilarity(queryEmbedding, embedding)
			similarities = append(similarities, sim)
		}
	}

	// Aggregate scores based on method
	return c.aggregateScores(similarities, rule)
}

// matchesLegacy uses the original runtime computation approach
func (c *EmbeddingClassifier) matchesLegacy(text string, rule config.EmbeddingRule) (bool, float32, error) {
	// Determine model type
	modelType := "auto"
	if testModel := os.Getenv("EMBEDDING_MODEL_OVERRIDE"); testModel != "" {
		modelType = testModel
		logging.Infof("Embedding model override from env: %s", modelType)
	}

	result, err := calculateSimilarityBatch(
		text,
		rule.Candidates,
		0,
		modelType,
		768,
	)
	if err != nil {
		return false, 0.0, fmt.Errorf("keyword-based embedding similarity classification: failed to calculate batch similarity: %w", err)
	}

	// Extract similarities
	similarities := make([]float32, len(result.Matches))
	for i, match := range result.Matches {
		similarities[i] = match.Similarity
	}

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
		return aggregatedScore >= rule.SimilarityThreshold, aggregatedScore, nil

	case config.AggregationMethodMax:
		var aggregatedScore float32
		for _, sim := range similarities {
			if sim > aggregatedScore {
				aggregatedScore = sim
			}
		}
		return aggregatedScore >= rule.SimilarityThreshold, aggregatedScore, nil

	case config.AggregationMethodAny:
		for _, sim := range similarities {
			if sim >= rule.SimilarityThreshold {
				return true, rule.SimilarityThreshold, nil
			}
		}
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
