package classification

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ClassifyDetailedMultimodal performs full label scoring on a non-text query
// (image or audio) against rules whose effective QueryModality matches.
// payload is a base64 string, a data-URI ("data:image/png;base64,..."), or
// a local file path; the same accepted forms as the underlying multimodal
// FFI helpers.
//
// Use this when the request carries an image or audio attachment that
// should be classified against text anchors in the shared multimodal space.
// For text queries, use ClassifyDetailed.
func (c *EmbeddingClassifier) ClassifyDetailedMultimodal(modality config.QueryModality, payload string) (*EmbeddingClassificationResult, error) {
	return c.classifyDetailedMultimodalWithCache(context.Background(), modality, payload, nil)
}

// classifyDetailedMultimodalWithCache is the cache-aware variant of
// ClassifyDetailedMultimodal. When cache is non-nil and the (payload,
// targetDim) pair matches an entry computed by another signal evaluator
// during the same EvaluateAllSignalsWithContext call, the embedding is
// reused instead of recomputed via FFI. A nil cache is equivalent to the
// pre-cache behavior.
func (c *EmbeddingClassifier) classifyDetailedMultimodalWithCache(ctx context.Context, modality config.QueryModality, payload string, cache *requestMediaEmbeddingCache) (*EmbeddingClassificationResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(c.rules) == 0 {
		return &EmbeddingClassificationResult{}, nil
	}
	if payload == "" {
		return nil, fmt.Errorf("embedding similarity classification: query payload must be provided")
	}

	effective := config.QueryModality(strings.ToLower(strings.TrimSpace(string(modality))))
	if effective == "" || effective == config.QueryModalityText {
		return nil, fmt.Errorf("ClassifyDetailedMultimodal: modality must be %q or %q (got %q); use ClassifyDetailed for text",
			config.QueryModalityImage, config.QueryModalityAudio, modality)
	}

	if effective != config.QueryModalityImage && effective != config.QueryModalityAudio {
		return nil, fmt.Errorf("unsupported query modality %q", modality)
	}

	startTime := time.Now()

	matchingRules := c.rulesByModality[effective]
	if len(matchingRules) == 0 {
		logging.Infof("No embedding rules configured for modality=%s (total rules: %d)",
			effective, len(c.rules))
		return &EmbeddingClassificationResult{}, nil
	}

	queryEmbedding, err := cache.resolveFor(c.provider, effective, payload, c.optimizationConfig.TargetDimension, func() ([]float32, error) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if effective == config.QueryModalityAudio {
			return embedding.Audio(ctx, c.provider, payload, c.optimizationConfig.TargetDimension)
		}
		return embedding.Image(ctx, c.provider, payload, c.optimizationConfig.TargetDimension)
	})
	if err != nil {
		return nil, fmt.Errorf("failed to compute multimodal query embedding (modality=%s): %w", effective, err)
	}

	if ctxErr := ctx.Err(); ctxErr != nil {
		return nil, ctxErr
	}
	logging.Infof("Computed multimodal query embedding (modality: %s, dimension: %d)",
		effective, len(queryEmbedding))

	if ensureErr := c.ensureCandidateEmbeddings(ctx); ensureErr != nil {
		return nil, ensureErr
	}

	scoredRules, err := c.scoreRulesSlice(queryEmbedding, matchingRules)
	if err != nil {
		return nil, err
	}
	matched := c.findAllMatchedRules(scoredRules)

	elapsed := time.Since(startTime)
	logging.Infof("ClassifyDetailedMultimodal(%s) completed in %v: %d rules matched out of %d",
		effective, elapsed, len(matched), len(matchingRules))

	return &EmbeddingClassificationResult{
		Scores:  scoredRules,
		Matches: matched,
	}, nil
}
