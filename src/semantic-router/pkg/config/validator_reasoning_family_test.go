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
							Type:        ReasoningFamilyTypeChatTemplateKwargs,
							Parameter:   "enable_thinking",
							Modes:       []string{ReasoningModeEnabled, ReasoningModeDisabled},
							DefaultMode: ReasoningModeEnabled,
						},
						"local-effort": {
							Type:        ReasoningFamilyTypeReasoningEffort,
							Parameter:   "reasoning_effort",
							Levels:      []string{"low", "medium", "high"},
							Modes:       []string{ReasoningModeEnabled},
							DefaultMode: ReasoningModeEnabled,
						},
						"flagged-effort": {
							Type:                ReasoningFamilyTypeReasoningEffort,
							Parameter:           "reasoning_effort",
							ActivationParameter: "enable_thinking",
							EffortFlags:         map[string]string{"low": "low_effort", "medium": "medium_effort"},
							Levels:              []string{"none", "low", "medium", "high"},
							Default:             "high",
							Modes:               []string{ReasoningModeEnabled, ReasoningModeDisabled},
							DefaultMode:         ReasoningModeEnabled,
							Disabled:            "none",
						},
						"mistral": {
							Type:        ReasoningFamilyTypeTopLevelReasoningEffort,
							Parameter:   "reasoning_effort",
							Levels:      []string{"high"},
							Modes:       []string{ReasoningModeEnabled, ReasoningModeDisabled},
							DefaultMode: ReasoningModeDisabled,
						},
					},
				},
			},
		}

		if err := validateReasoningFamilyContracts(cfg); err != nil {
			t.Fatalf("validateReasoningFamilyContracts() error = %v", err)
		}
	})

	t.Run("accepts an omitted operator mode contract", func(t *testing.T) {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				ReasoningConfig: ReasoningConfig{
					ReasoningFamilies: map[string]ReasoningFamilyConfig{
						"custom-toggle": {
							Type: ReasoningFamilyTypeChatTemplateKwargs, Parameter: "enable_thinking",
						},
					},
				},
			},
		}

		if err := validateReasoningFamilyContracts(cfg); err != nil {
			t.Fatalf("validateReasoningFamilyContracts() error = %v", err)
		}
	})
}

func TestValidateInvalidReasoningFamilyContracts(t *testing.T) {
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
		{
			name: "activation parameter cannot duplicate effort parameter",
			family: ReasoningFamilyConfig{
				Type:                ReasoningFamilyTypeReasoningEffort,
				Parameter:           "reasoning_effort",
				ActivationParameter: "reasoning_effort",
			},
			wantErr: "activation_parameter must differ from parameter",
		},
		{
			name: "effort family requires levels",
			family: ReasoningFamilyConfig{
				Type: ReasoningFamilyTypeReasoningEffort, Parameter: "reasoning_effort",
				Modes: []string{ReasoningModeEnabled}, DefaultMode: ReasoningModeEnabled,
			},
			wantErr: "levels must be set for effort-based reasoning",
		},
		{
			name: "effort flags require activation",
			family: ReasoningFamilyConfig{
				Type: ReasoningFamilyTypeReasoningEffort, Parameter: "reasoning_effort",
				EffortFlags: map[string]string{"low": "low_effort"},
				Levels:      []string{"low", "high"}, Modes: []string{ReasoningModeEnabled}, DefaultMode: ReasoningModeEnabled,
			},
			wantErr: "effort_flags requires activation_parameter",
		},
		{
			name: "effort flags must identify all but at most one level",
			family: ReasoningFamilyConfig{
				Type: ReasoningFamilyTypeReasoningEffort, Parameter: "reasoning_effort",
				ActivationParameter: "enable_thinking", EffortFlags: map[string]string{"low": "low_effort"},
				Levels: []string{"low", "medium", "high"}, Modes: []string{ReasoningModeEnabled}, DefaultMode: ReasoningModeEnabled,
			},
			wantErr: "multiple effort levels indistinguishable",
		},
		{
			name: "default mode requires modes",
			family: ReasoningFamilyConfig{
				Type: ReasoningFamilyTypeChatTemplateKwargs, Parameter: "thinking",
				DefaultMode: ReasoningModeEnabled,
			},
			wantErr: "modes must not be empty",
		},
		{
			name: "every family requires a default mode",
			family: ReasoningFamilyConfig{
				Type: ReasoningFamilyTypeChatTemplateKwargs, Parameter: "thinking",
				Modes: []string{ReasoningModeEnabled},
			},
			wantErr: "default_mode must not be empty",
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
