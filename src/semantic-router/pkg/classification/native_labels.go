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

// validateTokenLabels checks a prepared token binding against its consumer's
// mapping. A ready-made span question serves no label list, since its spans
// name their own labels, so its mapping must be empty too.
func validateTokenLabels(capability binding.Capability, declared []string) error {
	if capability.Preset == "" {
		return validateNativeLabelOrder(capability.Labels, declared, nil)
	}
	if len(declared) > 0 {
		return fmt.Errorf("%w: the %s question names its own labels; remove the consumer's label mapping", binding.ErrCapability, capability.Preset)
	}
	return nil
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
