package config

import (
	"strings"
	"testing"
)

func TestValidatePromptCachePluginContract(t *testing.T) {
	tests := []struct {
		name      string
		payload   map[string]interface{}
		wantError string
	}{
		{
			name:    "defaults",
			payload: map[string]interface{}{},
		},
		{
			name: "explicit contract",
			payload: map[string]interface{}{
				"enabled":        true,
				"ttl":            "1h",
				"targets":        []string{"instructions", "tools"},
				"on_unsupported": "reject",
			},
		},
		{
			name: "invalid ttl",
			payload: map[string]interface{}{
				"enabled": true,
				"ttl":     "10m",
			},
			wantError: "ttl must be 5m or 1h",
		},
		{
			name: "invalid target",
			payload: map[string]interface{}{
				"enabled": true,
				"targets": []string{"messages"},
			},
			wantError: "targets must contain only instructions or tools",
		},
		{
			name: "empty targets",
			payload: map[string]interface{}{
				"enabled": true,
				"targets": []string{},
			},
			wantError: "targets must not be empty",
		},
		{
			name: "duplicate target",
			payload: map[string]interface{}{
				"enabled": true,
				"targets": []string{"tools", "tools"},
			},
			wantError: "targets must not contain duplicates",
		},
		{
			name: "target with whitespace",
			payload: map[string]interface{}{
				"enabled": true,
				"targets": []string{" instructions"},
			},
			wantError: "targets must contain only instructions or tools",
		},
		{
			name: "invalid unsupported behavior",
			payload: map[string]interface{}{
				"enabled":        true,
				"on_unsupported": "ignore",
			},
			wantError: "on_unsupported must be skip or reject",
		},
		{
			name: "unknown field",
			payload: map[string]interface{}{
				"enabled": true,
				"typo":    true,
			},
			wantError: "unknown field \"typo\"",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateDecisionPluginPayload(
				"route",
				0,
				DecisionPlugin{
					Type:          DecisionPluginPromptCache,
					Configuration: MustStructuredPayload(test.payload),
				},
			)
			if test.wantError == "" {
				if err != nil {
					t.Fatalf("validate prompt cache plugin: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.wantError) {
				t.Fatalf("error = %v, want substring %q", err, test.wantError)
			}
		})
	}
}

func TestPromptCachePluginDefaults(t *testing.T) {
	got := (&PromptCachePluginConfig{}).withDefaults()

	if got.TTL != "5m" {
		t.Fatalf("TTL = %q, want 5m", got.TTL)
	}
	if got.OnUnsupported != PromptCacheUnsupportedSkip {
		t.Fatalf("OnUnsupported = %q, want %q", got.OnUnsupported, PromptCacheUnsupportedSkip)
	}
	if len(got.Targets) != 2 ||
		got.Targets[0] != PromptCacheTargetInstructions ||
		got.Targets[1] != PromptCacheTargetTools {
		t.Fatalf("Targets = %#v", got.Targets)
	}
}
