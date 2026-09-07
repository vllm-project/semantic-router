package evaluationplane

import (
	"strings"
	"testing"
)

func TestBrokerResponseContentUsesVisibleThenReasoningText(t *testing.T) {
	tests := []struct {
		name    string
		message map[string]any
		want    *string
	}{
		{
			name:    "visible content has priority",
			message: map[string]any{"content": "final answer", "reasoning": "private derivation"},
			want:    responseTextPointer("final answer"),
		},
		{
			name:    "reasoning attests a null content response",
			message: map[string]any{"content": nil, "reasoning": "bounded reasoning output"},
			want:    responseTextPointer("bounded reasoning output"),
		},
		{
			name:    "reasoning content alias is accepted",
			message: map[string]any{"content": nil, "reasoning_content": "provider reasoning output"},
			want:    responseTextPointer("provider reasoning output"),
		},
		{
			name:    "an explicit empty content remains the visible response",
			message: map[string]any{"content": "", "reasoning": "must not replace visible content"},
			want:    responseTextPointer(""),
		},
		{
			name:    "missing response text remains unattested",
			message: map[string]any{"content": nil},
			want:    nil,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			payload := map[string]any{
				"choices": []any{map[string]any{"message": test.message}},
			}
			got := brokerResponseContent(payload)
			if (got == nil) != (test.want == nil) ||
				(got != nil && test.want != nil && *got != *test.want) {
				t.Fatalf("broker response content = %v, want %v", got, test.want)
			}
		})
	}
}

func responseTextPointer(value string) *string {
	return &value
}

func TestBrokerObservedFieldsDoNotRetainUnboundedStrings(t *testing.T) {
	oversized := strings.Repeat("x", maxBrokerObservedFieldBytes+1)
	entry := executionAttestationEntry{}
	populateBrokerObservedFields(&entry, map[string]any{
		"selected_model":   oversized,
		"selection_status": oversized,
		"selection_method": oversized,
		"recipe":           oversized,
		"decision_result": map[string]any{
			"decision_name": oversized,
			"algorithm":     oversized,
		},
	})
	if entry.SelectedModel != nil || entry.SelectionStatus != nil || entry.SelectionMethod != nil ||
		entry.Recipe != nil || entry.DecisionName != nil || entry.Algorithm != nil {
		t.Fatalf("unbounded broker observations were retained: %+v", entry)
	}
}
