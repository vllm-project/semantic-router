package classification

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// compileTokenLabels copies the task's label declarations. Outside labels are
// explicit: label index zero can be an entity for tasks other than PII.
func compileTokenLabels(labels tasks.TokenLabelSet) (map[string]struct{}, map[string]struct{}, error) {
	known := make(map[string]struct{}, len(labels.Labels))
	outside := make(map[string]struct{}, len(labels.Outside))
	for _, label := range labels.Outside {
		label = stripBIOPrefix(label)
		if label == "" {
			return nil, nil, fmt.Errorf("token_spans outside labels must not be empty")
		}
		outside[label] = struct{}{}
	}
	for _, label := range labels.Labels {
		label = stripBIOPrefix(label)
		if label == "" {
			return nil, nil, fmt.Errorf("token_spans entity labels must not be empty")
		}
		if _, conflict := outside[label]; conflict {
			return nil, nil, fmt.Errorf("token_spans label %q is both an entity and outside label", label)
		}
		known[label] = struct{}{}
	}
	if len(known) == 0 {
		return nil, nil, fmt.Errorf("token_spans task label mapping is required")
	}
	return known, outside, nil
}

// stripBIOPrefix removes the BIO sequence labeling prefix from a token label.
// For example: "B-PERSON" → "PERSON", "I-DATE_TIME" → "DATE_TIME", "PERSON" → "PERSON".
func stripBIOPrefix(s string) string {
	if len(s) > 2 && s[1] == '-' {
		switch s[0] {
		case 'B', 'I', 'E':
			return s[2:]
		}
	}
	return s
}
