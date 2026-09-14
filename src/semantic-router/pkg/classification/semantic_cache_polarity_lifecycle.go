package classification

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
)

func (c *Classifier) needsSemanticCacheNLIForRuntime() bool {
	return c != nil && c.Config != nil && c.Config.NeedsLocalNLIForSemanticCache()
}

func (c *Classifier) initializeSemanticCacheNLI() error {
	models := c.models
	if models == nil {
		models = standaloneModelRuntime()
	}
	detector := &HallucinationDetector{models: models, nliConfig: &c.Config.HallucinationMitigation.NLIModel}
	if err := detector.InitializeNLI(); err != nil {
		return fmt.Errorf("prepare NLI for semantic cache polarity guard: %w", err)
	}
	c.polarityNLI = detector
	return nil
}

// PolarityVerifier borrows this classifier's immutable model binding. Its cache
// belongs to the same generation and drains before the binding is released.
func (c *Classifier) PolarityVerifier() cache.PolarityVerifyFunc {
	if c == nil || c.polarityNLI == nil {
		return nil
	}
	return func(ctx context.Context, cachedQuery, incomingQuery string) (float32, error) {
		result, err := c.polarityNLI.ClassifyNLI(ctx, cachedQuery, incomingQuery)
		if err != nil {
			return 0, err
		}
		return result.ContradictProb, nil
	}
}
