package config

import (
	"strings"
	"testing"
)

var contextCompressionValidationCases = []struct {
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
		name: "canonical_nested_contract",
		payload: map[string]interface{}{
			"enabled": true,
			"mode":    "auto",
			"budget": map[string]interface{}{
				"trigger_tokens":        "auto",
				"target_tokens":         "auto",
				"reserve_output_tokens": 4096,
			},
			"targets": map[string]interface{}{
				"tool_outputs": map[string]interface{}{
					"mode":          "extractive",
					"min_tokens":    2000,
					"target_tokens": 1000,
				},
				"history": map[string]interface{}{"mode": "preserve"},
				"rag":     map[string]interface{}{"mode": "preserve"},
			},
			"scoring": map[string]interface{}{"method": "bm25"},
			"request_controls": map[string]interface{}{
				"enabled":           true,
				"allowed":           []string{"bypass", "target"},
				"max_target_tokens": 8000,
			},
			"failure_mode": "fail_open",
		},
	},
	{
		name: "embedding_requires_model_ref",
		payload: map[string]interface{}{
			"enabled": true,
			"scoring": map[string]interface{}{"method": "hybrid"},
		},
		wantErr: "embedding_model_ref",
	},
	{
		name: "recovery_requires_shared_store",
		payload: map[string]interface{}{
			"enabled":  true,
			"recovery": map[string]interface{}{"enabled": true},
		},
		wantErr: "recovery.store",
	},
	{
		name: "target_equals_min",
		payload: map[string]interface{}{
			"enabled": true,
			"targets": map[string]interface{}{
				"tool_outputs": map[string]interface{}{
					"mode":          "extractive",
					"min_tokens":    1000,
					"target_tokens": 1000,
				},
			},
		},
		wantErr: "target_tokens must be less than min_tokens",
	},
	{
		name: "negative_threshold",
		payload: map[string]interface{}{
			"enabled": true,
			"targets": map[string]interface{}{
				"tool_outputs": map[string]interface{}{
					"mode":       "extractive",
					"min_tokens": -1,
				},
			},
		},
		wantErr: "cannot be negative",
	},
	{
		name: "flat_field_rejected",
		payload: map[string]interface{}{
			"enabled":    true,
			"min_tokens": 1000,
		},
		wantErr: "unknown field",
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

func TestValidateContextCompressionPluginContract(t *testing.T) {
	for _, test := range contextCompressionValidationCases {
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
