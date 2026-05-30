package classification

import (
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Classify performs Embedding similarity classification on the given text.
// Returns the single best matching rule. Wraps ClassifyAll internally.
func (c *EmbeddingClassifier) Classify(text string) (string, float64, error) {
	matched, err := c.ClassifyAll(text)
	if err != nil {
		return "", 0.0, err
	}
	if len(matched) == 0 {
		return "", 0.0, nil
	}
	best := matched[0]
	for _, m := range matched[1:] {
		if m.Score > best.Score {
			best = m
		}
	}
	return best.RuleName, best.Score, nil
}

// ClassifyAll performs embedding similarity classification on the given text.
// Returns the highest-ranking matched rules, limited by embedding_config.top_k
// (default 1, 0 disables truncation). When top_k is increased, the decision
// engine can compose multiple embedding matches together.
func (c *EmbeddingClassifier) ClassifyAll(text string) ([]MatchedRule, error) {
	result, err := c.ClassifyDetailed(text)
	if err != nil {
		return nil, err
	}
	return c.sortAndLimitMatches(result.Matches), nil
}

// ClassifyDetailed performs full label scoring on a TEXT query and returns
// the complete score distribution plus all accepted matches before top-k
// output shaping. Only rules whose effective QueryModality is "text"
// participate. For image/audio queries, use ClassifyDetailedMultimodal.
func (c *EmbeddingClassifier) ClassifyDetailed(text string) (*EmbeddingClassificationResult, error) {
	if len(c.rules) == 0 {
		return &EmbeddingClassificationResult{}, nil
	}
	if text == "" {
		return nil, fmt.Errorf("embedding similarity classification: query must be provided")
	}

	startTime := time.Now()

	textRules := c.rulesByModality[config.QueryModalityText]
	if len(textRules) == 0 {
		logging.Infof("No embedding rules configured for text-modality queries (text rules: %d / total: %d)",
			0, len(c.rules))
		return &EmbeddingClassificationResult{}, nil
	}

	modelType := c.getModelType()
	queryEmbedding, err := c.computeEmbedding(text, modelType)
	if err != nil {
		return nil, fmt.Errorf("failed to compute query embedding: %w", err)
	}

	logging.Infof("Computed query embedding (model: %s, dimension: %d)", modelType, len(queryEmbedding))

	if ensureErr := c.ensureCandidateEmbeddings(); ensureErr != nil {
		return nil, ensureErr
	}

	scoredRules, err := c.scoreRulesSlice(queryEmbedding, textRules)
	if err != nil {
		return nil, err
	}
	matched := c.findAllMatchedRules(scoredRules)

	elapsed := time.Since(startTime)
	logging.Infof("ClassifyDetailed completed in %v: %d rules matched out of %d (modality=text)",
		elapsed, len(matched), len(textRules))

	return &EmbeddingClassificationResult{
		Scores:  scoredRules,
		Matches: matched,
	}, nil
}
