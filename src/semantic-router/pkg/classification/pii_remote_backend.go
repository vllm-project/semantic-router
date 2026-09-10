package classification

import (
	"context"
	"errors"
	"fmt"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// piiHTTPBackend adapts the remote token classifier to PII's historical
// inference interface, so PIIDetected, PIIEntities, MatchedPIIRules and masking
// keep working unchanged whether the spans came from Candle or from a remote
// token_spans.v1 provider. The request context reaches the HTTP call, so an
// admission deadline or a cancelled request stops the remote call too.
type piiHTTPBackend struct {
	backend *HTTPTokenClassifierInference
}

func newPIIHTTPBackend(external *config.ExternalModelConfig, mapping *PIIMapping, deadline time.Duration) (PIIInference, error) {
	backend, err := newHTTPTokenClassifierInference(external, mapping, deadline)
	if err != nil {
		return nil, fmt.Errorf("failed to create PII token_spans backend: %w", err)
	}
	return &piiHTTPBackend{backend: backend}, nil
}

// ClassifyTokens returns the remote spans as a TokenClassificationResult. A
// partial response (ErrTokenSpansTruncated) keeps its spans and surfaces the
// error: signal evaluation counts the spans it did get and lets
// classifier.pii.on_error decide whether the unseen remainder blocks.
func (p *piiHTTPBackend) ClassifyTokens(ctx context.Context, text string) (candle_binding.TokenClassificationResult, error) {
	entities, err := p.backend.classifyTokens(ctx, text)
	if err != nil {
		if errors.Is(err, ErrTokenSpansTruncated) {
			return candle_binding.TokenClassificationResult{Entities: entities}, err
		}
		return candle_binding.TokenClassificationResult{}, err
	}
	return candle_binding.TokenClassificationResult{Entities: entities}, nil
}

// Close releases the remote connector so a retired classifier generation does
// not keep the previous backend's idle connections alive across reloads.
func (p *piiHTTPBackend) Close() error {
	if p.backend == nil {
		return nil
	}
	return p.backend.Close()
}
