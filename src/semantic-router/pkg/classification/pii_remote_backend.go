package classification

import (
	"context"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// piiHTTPBackend adapts the remote token classifier to PII's historical
// inference interface, so PIIDetected, PIIEntities, MatchedPIIRules and masking
// keep working unchanged whether the spans came from Candle or from a remote
// token_spans.v1 provider. The request context reaches the HTTP call, so an
// admission deadline or a cancelled request stops the remote call too.
type piiHTTPBackend struct {
	backend *HTTPTokenClassifierInference
}

func newPIIHTTPTokenClassifierInference(external *config.ExternalModelConfig, mapping *PIIMapping, deadline time.Duration) (*HTTPTokenClassifierInference, error) {
	if mapping == nil || len(mapping.LabelToIdx) == 0 {
		return nil, fmt.Errorf("PII label mapping is required for token_spans")
	}
	known, outside := knownPIILabels(mapping)
	labels := tasks.TokenLabelSet{}
	for label := range known {
		labels.Labels = append(labels.Labels, label)
	}
	for label := range outside {
		labels.Outside = append(labels.Outside, label)
	}
	return newHTTPTokenClassifierInference(external, labels, deadline)
}

func newPIIHTTPBackend(external *config.ExternalModelConfig, mapping *PIIMapping, deadline time.Duration) (PIIInference, error) {
	backend, err := newPIIHTTPTokenClassifierInference(external, mapping, deadline)
	if err != nil {
		return nil, fmt.Errorf("failed to create PII token_spans backend: %w", err)
	}
	return &piiHTTPBackend{backend: backend}, nil
}

// ClassifyTokens returns the remote spans as a TokenClassificationResult. A
// partial response (ErrTokenSpansTruncated) keeps its spans and surfaces the
// error: signal evaluation counts the spans it did get and lets
// classifier.pii.on_error decide whether the unseen remainder blocks.
func (p *piiHTTPBackend) ClassifyTokens(ctx context.Context, text string) (tasks.TokenClassificationResult, error) {
	return p.backend.ClassifyTokens(ctx, text)
}

// Close releases the remote connector so a retired classifier generation does
// not keep the previous backend's idle connections alive across reloads.
func (p *piiHTTPBackend) Close() error {
	if p.backend == nil {
		return nil
	}
	return p.backend.Close()
}

// knownPIILabels collects the mapping's entity label names with any BIO prefix
// removed, so a provider may say PERSON whether the mapping file was written
// as PERSON or B-PERSON. The outside labels are returned separately and are
// not part of the entity set: class zero of the mapping (the native
// classifier's no-entity class) and the literal "O".
func knownPIILabels(mapping *PIIMapping) (known, outside map[string]struct{}) {
	known = make(map[string]struct{})
	outside = map[string]struct{}{"O": {}}
	if mapping == nil {
		return known, outside
	}
	if zero, ok := mapping.IdxToLabel["0"]; ok && stripBIOPrefix(zero) != "" {
		outside[stripBIOPrefix(zero)] = struct{}{}
	}
	for label, idx := range mapping.LabelToIdx {
		if idx == 0 {
			outside[stripBIOPrefix(label)] = struct{}{}
		}
	}
	add := func(label string) {
		label = stripBIOPrefix(label)
		if _, isOutside := outside[label]; !isOutside && label != "" {
			known[label] = struct{}{}
		}
	}
	for label := range mapping.LabelToIdx {
		add(label)
	}
	for _, label := range mapping.IdxToLabel {
		add(label)
	}
	return known, outside
}
