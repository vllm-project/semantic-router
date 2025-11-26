package classification

import (
	"fmt"
	"os"

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
	var mostMatchedRule string
	for _, rule := range c.rules {
		matched, aggregatedScore, err := c.matches(text, rule) // Error handled
		if err != nil {
			return "", 0.0, err // Propagate error
		}
		if matched {
			if len(rule.Candidates) > 0 {
				logging.Infof("Keyword-based embedding similarity classification matched rule %q with candidates: %v, confidence score %s", rule.Name, rule.Candidates, aggregatedScore)
			} else {
				logging.Infof("Keyword-based embedding similarity classification do not match rule %q with candidates: %v, confidence score %s", rule.Name, rule.Candidates, aggregatedScore)
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
func (c *EmbeddingClassifier) matches(text string, rule config.EmbeddingRule) (bool, float32, error) {
	// Validate input
	if text == "" {
		return false, 0.0, fmt.Errorf("keyword-based embedding similarity classification: query must be provided")
	}
	if len(rule.Candidates) == 0 {
		return false, 0.0, fmt.Errorf("keyword-based embedding similarity classification: candidates must be provided")
	}

	// Determine model type: Check for test override via environment variable
	// This allows CI/tests to force a specific model (e.g., "qwen3") when Gemma isn't available
	// Production uses "auto" by default (respects Rust heuristic: Gemma for short texts, Qwen3 for long)
	modelType := "auto" // Default: use Rust auto-selection heuristic
	if testModel := os.Getenv("EMBEDDING_MODEL_OVERRIDE"); testModel != "" {
		modelType = testModel
		logging.Infof("Embedding model override from env: %s", modelType)
	}

	result, err := calculateSimilarityBatch(
		text,
		rule.Candidates,
		0,         // return scores for all the candidates
		modelType, // use model type (auto or override)
		768,       // use default dimension
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
