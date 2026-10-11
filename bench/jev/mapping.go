package main

import (
	"fmt"
	"math"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// orderedDistribution is a research-only conversion to the Router result type.
// labelOrder must come from the consumer's configuration, never map iteration.
// Existing classification.alignScoresToMapping is package-private; this helper
// does not register a production adapter or alter that implementation.
func orderedDistribution(a answer, labelOrder []string) (tasks.LabelDistribution, error) {
	if len(labelOrder) == 0 || len(labelOrder) != len(a.Probabilities) {
		return tasks.LabelDistribution{}, fmt.Errorf("label set mismatch")
	}
	values := make([]float32, len(labelOrder))
	seen := make(map[string]bool, len(labelOrder))
	var sum float64
	for i, label := range labelOrder {
		p, present := a.Probabilities[label]
		if label == "" || seen[label] || !present || p == nil || !probability(*p) {
			return tasks.LabelDistribution{}, fmt.Errorf("duplicate, missing, or invalid label probability")
		}
		seen[label] = true
		values[i] = float32(*p)
		sum += float64(values[i])
	}
	// Check the representation actually returned to the Router. Conversion to
	// float32 may round values; it must not silently repair or normalize mass.
	if math.Abs(sum-1) > 1e-3 {
		return tasks.LabelDistribution{}, fmt.Errorf("converted probabilities do not sum to one")
	}
	return tasks.LabelDistribution{Probabilities: values}, nil
}
