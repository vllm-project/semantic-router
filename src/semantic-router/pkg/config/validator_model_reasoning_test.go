package config

import (
	"strings"
	"testing"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

type modelReasoningValidationCase struct {
	name     string
	cfg      *RouterConfig
	modelRef ModelRef
	wantErr  string
}

func modelReasoningBool(value bool) *bool { return &value }

func configWithModelReasoningFamily(family ReasoningFamilyConfig) *RouterConfig {
	cfg := &RouterConfig{}
	cfg.ReasoningConfig = ReasoningConfig{
		ReasoningFamilies: map[string]ReasoningFamilyConfig{"test": family},
	}
	cfg.ModelConfig = map[string]ModelParams{"model": {ReasoningFamily: "test"}}
	return cfg
}

func TestValidateModelRefReasoningControl(t *testing.T) {
	tests := []modelReasoningValidationCase{
		{
			name: "custom model without family can leave reasoning disabled",
			cfg:  &RouterConfig{},
			modelRef: ModelRef{Model: "custom", ModelReasoningControl: ModelReasoningControl{
				UseReasoning: modelReasoningBool(false),
			}},
		},
		{
			name: "custom model without family preserves legacy controls",
			cfg:  &RouterConfig{},
			modelRef: ModelRef{Model: "custom", ModelReasoningControl: ModelReasoningControl{
				UseReasoning: modelReasoningBool(true), ReasoningEffort: "operator-defined",
			}},
		},
		{
			name: "custom model without family cannot use projected mode",
			cfg: func() *RouterConfig {
				cfg := &RouterConfig{}
				cfg.ModelConfig = map[string]ModelParams{"custom": {}}
				cfg.EffectiveModelRegistry = &modelcatalog.EffectiveRegistry{}
				return cfg
			}(),
			modelRef: ModelRef{Model: "custom", ModelReasoningControl: ModelReasoningControl{
				UseReasoning: modelReasoningBool(true), ReasoningMode: ReasoningModeAdaptive,
			}},
			wantErr: "reasoning_mode requires a reasoning family",
		},
		{
			name: "detached routing fragment defers mode projection validation",
			cfg:  &RouterConfig{},
			modelRef: ModelRef{Model: "custom", ModelReasoningControl: ModelReasoningControl{
				UseReasoning: modelReasoningBool(true), ReasoningMode: ReasoningModeAdaptive,
			}},
		},
		{
			name: "always-on family rejects disable",
			cfg: configWithModelReasoningFamily(ReasoningFamilyConfig{
				Type: ReasoningFamilyTypeReasoningEffort, Parameter: "reasoning_effort",
				Levels: []string{"low", "high"}, Default: "high", Modes: []string{ReasoningModeEnabled},
			}),
			modelRef: ModelRef{Model: "model", ModelReasoningControl: ModelReasoningControl{
				UseReasoning: modelReasoningBool(false),
			}},
			wantErr: "always-on reasoning family",
		},
		{
			name: "disabled mode rejects effort",
			cfg: configWithModelReasoningFamily(ReasoningFamilyConfig{
				Type: ReasoningFamilyTypeReasoningEffort, Parameter: "reasoning_effort",
				Levels: []string{"low", "high"}, Default: "high", Disabled: "none",
				Modes: []string{ReasoningModeEnabled, ReasoningModeDisabled},
			}),
			modelRef: ModelRef{Model: "model", ModelReasoningControl: ModelReasoningControl{
				UseReasoning: modelReasoningBool(false), ReasoningMode: ReasoningModeDisabled, ReasoningEffort: "low",
			}},
			wantErr: "cannot be set while reasoning is disabled",
		},
	}
	runModelReasoningValidationCases(t, tests)
}

func TestValidateModelRefReasoningProviderProjection(t *testing.T) {
	tests := []modelReasoningValidationCase{
		{
			name: "adaptive family accepts a supported effort",
			cfg: configWithModelReasoningFamily(ReasoningFamilyConfig{
				Type: ReasoningFamilyTypeReasoningEffort, Parameter: "effort",
				Levels: []string{"low", "high"}, Default: "high",
				Modes: []string{ReasoningModeAdaptive, ReasoningModeDisabled},
			}),
			modelRef: ModelRef{Model: "model", ModelReasoningControl: ModelReasoningControl{
				UseReasoning: modelReasoningBool(true), ReasoningMode: ReasoningModeAdaptive, ReasoningEffort: "low",
			}},
		},
		{
			name: "boolean family rejects fake effort",
			cfg: configWithModelReasoningFamily(ReasoningFamilyConfig{
				Type: ReasoningFamilyTypeChatTemplateKwargs, Parameter: "enable_thinking",
				Modes: []string{ReasoningModeEnabled, ReasoningModeDisabled}, DefaultMode: ReasoningModeEnabled,
			}),
			modelRef: ModelRef{Model: "model", ModelReasoningControl: ModelReasoningControl{
				UseReasoning: modelReasoningBool(true), ReasoningEffort: "enabled",
			}},
			wantErr: "mode-only reasoning family",
		},
		{
			name: "provider binding can narrow model modes",
			cfg: func() *RouterConfig {
				cfg := configWithModelReasoningFamily(ReasoningFamilyConfig{
					Type: ReasoningFamilyTypeReasoningMode, Parameter: "thinking_mode",
					Modes:       []string{ReasoningModeEnabled, ReasoningModeDisabled, ReasoningModeAdaptive},
					DefaultMode: ReasoningModeAdaptive,
				})
				cfg.ModelConfig["model"] = ModelParams{
					ReasoningFamily: "test", PreferredEndpoints: []string{"minimax-primary"},
				}
				cfg.ProviderProfiles = map[string]ProviderProfile{
					"minimax-primary": {Type: "minimax", ReasoningModes: []string{ReasoningModeDisabled, ReasoningModeAdaptive}},
				}
				return cfg
			}(),
			modelRef: ModelRef{Model: "model", ModelReasoningControl: ModelReasoningControl{
				UseReasoning: modelReasoningBool(true), ReasoningMode: ReasoningModeEnabled,
			}},
			wantErr: "is not supported by provider",
		},
		{
			name: "provider transport cannot silently discard an effort ladder",
			cfg: func() *RouterConfig {
				cfg := configWithModelReasoningFamily(ReasoningFamilyConfig{
					Type: ReasoningFamilyTypeReasoningEffort, Parameter: "reasoning_effort",
					Levels: []string{"low", "high"}, Default: "high",
					Modes: []string{ReasoningModeEnabled}, DefaultMode: ReasoningModeEnabled,
				})
				cfg.ModelConfig["model"] = ModelParams{
					ReasoningFamily: "test", PreferredEndpoints: []string{"hosted"},
				}
				cfg.ProviderProfiles = map[string]ProviderProfile{
					"hosted": {
						Type: "minimax", ReasoningTransport: modelcatalog.ReasoningTransportThinkingObject,
					},
				}
				return cfg
			}(),
			modelRef: ModelRef{Model: "model", ModelReasoningControl: ModelReasoningControl{
				UseReasoning: modelReasoningBool(true), ReasoningEffort: "high",
			}},
			wantErr: "cannot project family type",
		},
	}
	runModelReasoningValidationCases(t, tests)
}

func runModelReasoningValidationCases(t *testing.T, tests []modelReasoningValidationCase) {
	t.Helper()
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateModelRefReasoningControl(test.cfg, "test", 0, test.modelRef)
			if test.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("error = %v, want substring %q", err, test.wantErr)
			}
		})
	}
}
