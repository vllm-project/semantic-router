package config

import (
	"strings"
	"testing"
)

func TestValidateReasoningFamilyContracts(t *testing.T) {
	t.Run("accepts every supported syntax", func(t *testing.T) {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				ReasoningConfig: ReasoningConfig{
					ReasoningFamilies: map[string]ReasoningFamilyConfig{
						"qwen": {
							Type:      ReasoningFamilyTypeChatTemplateKwargs,
							Parameter: "enable_thinking",
						},
						"local-effort": {
							Type:      ReasoningFamilyTypeReasoningEffort,
							Parameter: "reasoning_effort",
						},
						"mistral": {
							Type:      ReasoningFamilyTypeTopLevelReasoningEffort,
							Parameter: "reasoning_effort",
						},
					},
				},
			},
		}

		if err := validateReasoningFamilyContracts(cfg); err != nil {
			t.Fatalf("validateReasoningFamilyContracts() error = %v", err)
		}
	})

	tests := []struct {
		name    string
		family  ReasoningFamilyConfig
		wantErr string
	}{
		{
			name:    "unknown type",
			family:  ReasoningFamilyConfig{Type: "custom", Parameter: "reasoning_effort"},
			wantErr: "unsupported value",
		},
		{
			name:    "empty parameter",
			family:  ReasoningFamilyConfig{Type: ReasoningFamilyTypeReasoningEffort},
			wantErr: "parameter must not be empty",
		},
		{
			name: "top-level type uses canonical field",
			family: ReasoningFamilyConfig{
				Type:      ReasoningFamilyTypeTopLevelReasoningEffort,
				Parameter: "effort",
			},
			wantErr: `parameter must be "reasoning_effort"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &RouterConfig{
				IntelligentRouting: IntelligentRouting{
					ReasoningConfig: ReasoningConfig{
						ReasoningFamilies: map[string]ReasoningFamilyConfig{"test": tt.family},
					},
				},
			}
			err := validateReasoningFamilyContracts(cfg)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("validateReasoningFamilyContracts() error = %v, want substring %q", err, tt.wantErr)
			}
		})
	}
}
