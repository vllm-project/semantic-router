package classification

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// Generic token tasks retain outside labels. Only the PII consumer removes
// them, after resolving this recipe's mapping and checking score availability.
// A text the request already classified (PIIScans) is not classified again.
func (c *Classifier) classifyPIITokens(ctx context.Context, text string) (tasks.TokenClassificationResult, error) {
	scans := piiScansFrom(ctx)
	if result, ok := scans.lookup(text); ok {
		return result, nil
	}
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
	if err == nil {
		scans.store(text, result)
	}
	return result, err
}

// PIIScans keeps one request's complete PII token classifications by the
// hash of the text each read, so masking the provider-bound request asks the
// model only about text routing did not already read.
type PIIScans struct {
	mu     sync.Mutex
	byText map[[sha256.Size]byte]tasks.TokenClassificationResult
}

type piiScansKey struct{}

// WithPIIScans makes the classifications under ctx reuse scans.
func WithPIIScans(ctx context.Context, scans *PIIScans) context.Context {
	return context.WithValue(ctx, piiScansKey{}, scans)
}

func piiScansFrom(ctx context.Context) *PIIScans {
	if ctx == nil {
		return nil
	}
	scans, _ := ctx.Value(piiScansKey{}).(*PIIScans)
	return scans
}

func (s *PIIScans) lookup(text string) (tasks.TokenClassificationResult, bool) {
	if s == nil {
		return tasks.TokenClassificationResult{}, false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	result, ok := s.byText[sha256.Sum256([]byte(text))]
	return result, ok
}

func (s *PIIScans) store(text string, result tasks.TokenClassificationResult) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.byText == nil {
		s.byText = map[[sha256.Size]byte]tasks.TokenClassificationResult{}
	}
	s.byText[sha256.Sum256([]byte(text))] = result
}
