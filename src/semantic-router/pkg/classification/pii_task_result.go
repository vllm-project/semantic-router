package classification

import (
	"context"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// Generic token tasks retain outside labels. Only the PII consumer removes
// them, after resolving this recipe's mapping and checking score availability.
func (c *Classifier) classifyPIITokens(ctx context.Context, text string) (tasks.TokenClassificationResult, error) {
	result, err := c.piiInference.ClassifyTokens(ctx, text)
	if err != nil && !errors.Is(err, tasks.ErrTokenSpansTruncated) {
		return result, err
	}
	if !result.HasScores() {
		return tasks.TokenClassificationResult{}, fmt.Errorf("PII threshold requires token probabilities: %w", tasks.ErrProbabilitiesUnavailable)
	}
	_, outside := knownPIILabels(c.PIIMapping)
	entities := make([]tasks.TokenEntity, 0, len(result.Entities))
	for _, entity := range result.Entities {
		if _, skip := outside[c.PIIMapping.TranslatePIIType(entity.EntityType)]; !skip {
			entities = append(entities, entity)
		}
	}
	result.Entities = entities
	return result, err
}
