package config

import (
	"strings"
	"testing"
)

func TestValidateContextCompressionPluginContract(t *testing.T) {
	tests := []struct {
		name    string
		payload map[string]interface{}
		wantErr string
	}{
		{
			name: "defaults",
			payload: map[string]interface{}{
				"enabled": true,
			},
		},
		{
			name: "explicit_budget",
			payload: map[string]interface{}{
				"enabled":       true,
				"min_tokens":    2000,
				"target_tokens": 1000,
				"compress_rag":  true,
			},
		},
		{
			name: "target_equals_min",
			payload: map[string]interface{}{
				"enabled":       true,
				"min_tokens":    1000,
				"target_tokens": 1000,
			},
			wantErr: "target_tokens must be less than min_tokens",
		},
		{
			name: "negative_threshold",
			payload: map[string]interface{}{
				"enabled":    true,
				"min_tokens": -1,
			},
			wantErr: "cannot be negative",
		},
		{
			name: "unknown_field",
			payload: map[string]interface{}{
				"enabled": true,
				"unsafe":  true,
			},
			wantErr: "unknown field",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateDecisionPluginPayload(
				"route",
				0,
				DecisionPlugin{
					Type:          DecisionPluginContextCompression,
					Configuration: MustStructuredPayload(test.payload),
				},
			)
			if test.wantErr == "" {
				if err != nil {
					t.Fatalf("valid context compression config rejected: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("expected error containing %q, got %v", test.wantErr, err)
			}
		})
	}
}
