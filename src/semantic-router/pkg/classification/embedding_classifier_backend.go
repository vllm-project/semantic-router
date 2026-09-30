package classification

import (
	"context"
	"fmt"
	"time"

	embeddingprovider "github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (c *Classifier) IsKeywordEmbeddingClassifierEnabled() bool {
	return len(c.Config.EmbeddingRules) > 0
}

func (c *Classifier) initializeKeywordEmbeddingClassifier() error {
	if !c.IsKeywordEmbeddingClassifierEnabled() || c.keywordEmbeddingClassifier == nil {
		return fmt.Errorf("keyword embedding similarity match is not properly configured")
	}
	if c.keywordEmbeddingClassifier.provider == nil {
		return fmt.Errorf("keyword embedding provider was not prepared")
	}
	return c.keywordEmbeddingClassifier.WarmupCandidateEmbeddings()
}

func (c *EmbeddingClassifier) inferenceBackend() string {
	if c.provider == nil {
		return "unprepared"
	}
	return c.provider.Backend()
}

func (c *EmbeddingClassifier) computeEmbedding(ctx context.Context, text string, modelType string, phases ...string) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	start := time.Now()
	vector, err := embeddingprovider.Embed(ctx, c.provider, text, embeddingprovider.Options{
		Dimension: c.optimizationConfig.TargetDimension, Layer: c.optimizationConfig.TargetLayer,
	})
	phase := "request"
	if len(phases) > 0 {
		phase = phases[0]
	}
	logging.Infof("[Perf] embedding inference (phase=%s, backend=%s, model=%s, dim=%d): %.3fms",
		phase, c.inferenceBackend(), modelType, len(vector), float64(time.Since(start).Microseconds())/1000.0)
	if err == nil {
		err = ctx.Err()
	}
	return vector, err
}
