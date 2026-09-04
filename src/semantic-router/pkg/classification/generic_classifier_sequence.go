package classification

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// declaredLabelMapping indexes a rule's labels by their declared position.
// Jailbreak and category read their mapping from a model-shipped file; a
// generic signal has none, so the config's own ordering is the contract.
type declaredLabelMapping struct {
	labels     []string
	labelToIdx map[string]int
}

func newDeclaredLabelMapping(labels []string) *declaredLabelMapping {
	labelToIdx := make(map[string]int, len(labels))
	for index, label := range labels {
		labelToIdx[label] = index
	}
	return &declaredLabelMapping{
		labels:     append([]string(nil), labels...),
		labelToIdx: labelToIdx,
	}
}

func (m *declaredLabelMapping) IndexForLabel(label string) (int, bool) {
	index, ok := m.labelToIdx[label]
	return index, ok
}

func (m *declaredLabelMapping) LabelFromIndex(classIndex int) (string, bool) {
	if classIndex < 0 || classIndex >= len(m.labels) {
		return "", false
	}
	return m.labels[classIndex], true
}

func (m *declaredLabelMapping) LabelCount() int {
	return len(m.labels)
}

// sequenceLabelClassifier lets a rule reuse the backends jailbreak and
// category already use, instead of a parallel remote path.
type sequenceLabelClassifier struct {
	backend *HTTPClassifierInference
	labels  []string
}

func newSequenceLabelClassifier(
	rule config.ClassifierSignalRule,
	external *config.ExternalModelConfig,
) (labelClassifier, error) {
	if external == nil {
		return nil, fmt.Errorf("external model %q is not configured", rule.Model)
	}
	backend, err := NewHTTPClassifierInference(external, newDeclaredLabelMapping(rule.Labels))
	if err != nil {
		return nil, err
	}
	return &sequenceLabelClassifier{
		backend: backend,
		labels:  append([]string(nil), rule.Labels...),
	}, nil
}

func (c *sequenceLabelClassifier) Classify(
	ctx context.Context,
	input string,
) (labelClassification, error) {
	result, err := c.backend.Classify(ctx, input)
	if err != nil {
		return labelClassification{}, err
	}
	scores := make(map[string]float64, len(c.labels))
	for index, label := range c.labels {
		scores[label] = float64(result.Probabilities[index])
	}
	return labelClassification{Scores: scores}, nil
}
