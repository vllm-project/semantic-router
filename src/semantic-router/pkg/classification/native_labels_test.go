package classification

import (
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

func TestNativeConsumerLabelOrder(t *testing.T) {
	for _, test := range []struct {
		name             string
		actual, declared []string
		normalize        func(string) string
		invalid          bool
	}{
		{name: "semantic order", actual: []string{"chat", "billing"}, declared: []string{"chat", "billing"}},
		{name: "swapped semantic order", actual: []string{"chat", "billing"}, declared: []string{"billing", "chat"}, invalid: true},
		{name: "explicit unnamed indices", actual: []string{"LABEL_0", "LABEL_1"}, declared: []string{"chat", "billing"}},
		{name: "mixed semantic and unnamed", actual: []string{"chat", "LABEL_1"}, declared: []string{"chat", "billing"}, invalid: true},
		{name: "PII wrong entity", actual: []string{"O", "B-SECRET"}, declared: []string{"O", "B-PERSON"}, invalid: true},
		{name: "PII wrong outside", actual: []string{"O", "B-SECRET"}, declared: []string{"B-SECRET", "O"}, invalid: true},
		{name: "existing feedback alias", actual: []string{"SAT", "WRONG_ANSWER"}, declared: []string{"satisfied", "wrong_answer"}, normalize: normalizeFeedbackLabel},
		{name: "no implicit generic alias", actual: []string{"SAT", "WRONG_ANSWER"}, declared: []string{"satisfied", "wrong_answer"}, invalid: true},
		{name: "missing index", actual: []string{"chat", "billing"}, declared: []string{"chat", ""}, invalid: true},
		{name: "duplicate declaration", actual: []string{"LABEL_0", "LABEL_1"}, declared: []string{"chat", "chat"}, invalid: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := validateNativeLabelOrder(test.actual, test.declared, test.normalize)
			if test.invalid != errors.Is(err, binding.ErrCapability) {
				t.Fatalf("label capability error = %v, invalid = %v", err, test.invalid)
			}
		})
	}
}
