package classification

import (
	"fmt"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// calculateSimilarityBatch is a package-level variable that points to the
// actual implementation in the candle_binding package. It exists so tests can
// override it.
var calculateSimilarityBatch = candle_binding.CalculateSimilarityBatch

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
	logging.Infof("Initialized KeywordEmbedding classifier with qwen3 model path %q and gemma model path %s", qwen3ModelPath, gemmaModelPath)
	return nil
}

// createEmbeddingInitializer creates the appropriate keyword embedding initializer based on configuration
func createEmbeddingInitializer() EmbeddingClassifierInitializer {
	return &ExternalModelBasedEmbeddingInitializer{}
}

type EmbeddingClassifier struct {
	rules []config.EmbeddingRule
}

// NewKeywordClassifier creates a new KeywordEmbeddingClassifier.
func NewEmbeddingClassifier(cfgRules []config.EmbeddingRule) (*EmbeddingClassifier, error) {
	return &EmbeddingClassifier{rules: cfgRules}, nil
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
	return c.keywordEmbeddingInitializer.Init(c.Config.InlineModels.Qwen3ModelPath, c.Config.InlineModels.GemmaModelPath, c.Config.InlineModels.EmbeddingModels.UseCPU)
}

// Classify performs keyword-based embedding similarity classification on the given text.
func (c *EmbeddingClassifier) Classify(text string) (string, float64, error) {
	var bestScore float32
	var mostMatchedCategory string
	for _, rule := range c.rules {
		matched, aggregatedScore, err := c.matches(text, rule) // Error handled
		if err != nil {
			return "", 0.0, err // Propagate error
		}
		if matched {
			if len(rule.Keywords) > 0 {
				logging.Infof("Keyword-based embedding similarity classification matched category %q with keywords: %v, confidence score %s", rule.Category, rule.Keywords, aggregatedScore)
			} else {
				logging.Infof("Keyword-based embedding similarity classification do not match category %q with keywords: %v, confidence score %s", rule.Category, rule.Keywords, aggregatedScore)
			}
			if aggregatedScore > bestScore {
				bestScore = aggregatedScore
				mostMatchedCategory = rule.Category
			}
		}
	}
	return mostMatchedCategory, float64(bestScore), nil
}

// matches checks if the text matches the given keyword rule.
func (c *EmbeddingClassifier) matches(text string, rule config.EmbeddingRule) (bool, float32, error) {
	// Validate input
	if text == "" {
		return false, 0.0, fmt.Errorf("keyword-based embedding similarity classification: query must be provided")
	}
	if len(rule.Keywords) == 0 {
		return false, 0.0, fmt.Errorf("keyword-based embedding similarity classification: keywords must be provided")
	}
	// Set defaults
	if rule.Dimension == 0 {
		rule.Dimension = 768 // Default to full dimension
	}
	if rule.Model == "auto" && rule.QualityPriority == 0 && rule.LatencyPriority == 0 {
		rule.QualityPriority = 0.5
		rule.LatencyPriority = 0.5
	}

	// Validate dimension
	validDimensions := map[int]bool{128: true, 256: true, 512: true, 768: true, 1024: true}
	if !validDimensions[rule.Dimension] {
		return false, 0.0, fmt.Errorf("keyword-based embedding similarity classification: dimension must be one of: 128, 256, 512, 768, 1024 (got %d)", rule.Dimension)
	}
	// Calculate batch similarity
	result, err := calculateSimilarityBatch(
		text,
		rule.Keywords,
		0, // return scores for all the keywords
		rule.Model,
		rule.Dimension,
	)
	if err != nil {
		return false, 0.0, fmt.Errorf("keyword-based embedding similarity classification: failed to calculate batch similarity: %w", err)
	}
	// Check for matches based on the aggregation method
	switch rule.AggregationMethodConfiged {
	case config.AggregationMethodMean:
		var aggregatedScore float32
		for _, match := range result.Matches {
			aggregatedScore += match.Similarity
		}
		aggregatedScore /= float32(len(result.Matches))
		if aggregatedScore >= rule.SimilarityThreshold {
			return true, aggregatedScore, nil
		} else {
			return false, aggregatedScore, nil
		}
	case config.AggregationMethodMax:
		var aggregatedScore float32
		for _, match := range result.Matches {
			if match.Similarity > aggregatedScore {
				aggregatedScore = match.Similarity
			}
		}
		if aggregatedScore >= rule.SimilarityThreshold {
			return true, aggregatedScore, nil
		} else {
			return false, aggregatedScore, nil
		}
	case config.AggregationMethodAny:
		for _, match := range result.Matches {
			if match.Similarity >= rule.SimilarityThreshold {
				return true, rule.SimilarityThreshold, nil
			}
		}
		return false, 0.0, nil

	}
	return false, 0.0, fmt.Errorf("keyword-based embedding similarity classification: unsupported keyword rule aggregation method: %q", rule.AggregationMethodConfiged)
}
