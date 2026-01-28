package classification

import (
	"fmt"
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
func (c *ComplexityClassifier) preloadCandidateEmbeddings() error {
	startTime := time.Now()
	totalEmbeddings := 0

	for _, rule := range c.rules {
		// Initialize maps for this rule
		c.hardEmbeddings[rule.Name] = make(map[string][]float32)
		c.easyEmbeddings[rule.Name] = make(map[string][]float32)

		// Precompute hard candidate embeddings
		for _, candidate := range rule.Hard.Candidates {
			output, err := getEmbeddingWithModelType(candidate, c.modelType, 0)
			if err != nil {
				return fmt.Errorf("failed to compute embedding for hard candidate '%s': %w", candidate, err)
			}
			c.hardEmbeddings[rule.Name][candidate] = output.Embedding
			totalEmbeddings++
		}

		// Precompute easy candidate embeddings
		for _, candidate := range rule.Easy.Candidates {
			output, err := getEmbeddingWithModelType(candidate, c.modelType, 0)
			if err != nil {
				return fmt.Errorf("failed to compute embedding for easy candidate '%s': %w", candidate, err)
			}
			c.easyEmbeddings[rule.Name][candidate] = output.Embedding
			totalEmbeddings++
		}
	}

	elapsed := time.Since(startTime)
	logging.Infof("Preloaded %d complexity embeddings (hard/easy candidates) in %v", totalEmbeddings, elapsed)

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
