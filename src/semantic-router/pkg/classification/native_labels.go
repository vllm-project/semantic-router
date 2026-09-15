package classification

import (
	"fmt"
	"strconv"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

func indexedNativeLabels(labels map[string]string) []string {
	ordered := make([]string, len(labels))
	for index := range ordered {
		ordered[index] = labels[strconv.Itoa(index)]
	}
	return ordered
}

// Native distributions use the model's output order. A sidecar may name an
// otherwise unnamed LABEL_n output, but must never relabel semantic outputs.
// Only consumers with an existing alias contract supply a normalizer.
func validateNativeLabelOrder(actual, declared []string, normalize func(string) string) error {
	if len(actual) == 0 || len(actual) != len(declared) {
		return fmt.Errorf("%w: prepared labels do not match consumer mapping: model=%d mapping=%d", binding.ErrCapability, len(actual), len(declared))
	}
	indexed := true
	for index, label := range actual {
		indexed = indexed && label == "LABEL_"+strconv.Itoa(index)
	}
	seen := make(map[string]bool, len(declared))
	for index, label := range declared {
		modelLabel := actual[index]
		if normalize != nil {
			label, modelLabel = normalize(label), normalize(modelLabel)
		}
		if label == "" || seen[label] {
			return fmt.Errorf("%w: consumer label mapping must be a contiguous bijection", binding.ErrCapability)
		}
		seen[label] = true
		if !indexed && modelLabel != label {
			return fmt.Errorf("%w: model label %q at index %d disagrees with consumer label %q", binding.ErrCapability, actual[index], index, declared[index])
		}
	}
	return nil
}
