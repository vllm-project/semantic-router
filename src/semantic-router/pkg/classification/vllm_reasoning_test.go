package classification

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestExternalModelReasoningRequestProjection(t *testing.T) {
	enabled := true
	disabled := false
	tests := []struct {
		name       string
		reasoning  *config.ExternalModelReasoningConfig
		family     config.ReasoningFamilyConfig
		wantEffort string
		wantChat   map[string]interface{}
	}{
		{name: "omitted"},
		{
			name:      "chat template disabled",
			reasoning: &config.ExternalModelReasoningConfig{Family: "test", UseReasoning: &disabled},
			family: config.ReasoningFamilyConfig{
				Type: config.ReasoningFamilyTypeChatTemplateKwargs, Parameter: "enable_thinking",
				Modes: []string{config.ReasoningModeEnabled, config.ReasoningModeDisabled},
			},
			wantChat: map[string]interface{}{"enable_thinking": false},
		},
		{
			name:      "default effort",
			reasoning: &config.ExternalModelReasoningConfig{Family: "test", UseReasoning: &enabled},
			family: config.ReasoningFamilyConfig{
				Type: config.ReasoningFamilyTypeReasoningEffort, Parameter: "reasoning_effort",
				Levels: []string{"low", "high"}, Default: "high",
			},
			wantChat: map[string]interface{}{"reasoning_effort": "high"},
		},
		{
			name:      "explicit effort",
			reasoning: &config.ExternalModelReasoningConfig{Family: "test", UseReasoning: &enabled, ReasoningEffort: "low"},
			family: config.ReasoningFamilyConfig{
				Type: config.ReasoningFamilyTypeReasoningEffort, Parameter: "reasoning_effort",
				Levels: []string{"low", "high"}, Default: "high",
			},
			wantChat: map[string]interface{}{"reasoning_effort": "low"},
		},
		{
			name:      "top-level effort",
			reasoning: &config.ExternalModelReasoningConfig{Family: "test", UseReasoning: &enabled, ReasoningEffort: "high"},
			family: config.ReasoningFamilyConfig{
				Type: config.ReasoningFamilyTypeTopLevelReasoningEffort, Parameter: "reasoning_effort",
				Levels: []string{"low", "high"}, Default: "low",
			},
			wantEffort: "high",
		},
		{
			name:      "activation and effort",
			reasoning: &config.ExternalModelReasoningConfig{Family: "test", UseReasoning: &enabled, ReasoningEffort: "low"},
			family: config.ReasoningFamilyConfig{
				Type: config.ReasoningFamilyTypeReasoningEffort, Parameter: "reasoning_effort", ActivationParameter: "enable_thinking",
				Levels: []string{"low", "high"}, Default: "high",
			},
			wantChat: map[string]interface{}{"reasoning_effort": "low", "enable_thinking": true},
		},
		{
			name:      "effort flags",
			reasoning: &config.ExternalModelReasoningConfig{Family: "test", UseReasoning: &enabled, ReasoningEffort: "low"},
			family: config.ReasoningFamilyConfig{
				Type: config.ReasoningFamilyTypeReasoningEffort, Parameter: "reasoning_effort", ActivationParameter: "enable_thinking",
				Levels: []string{"low", "high"}, Default: "high", EffortFlags: map[string]string{"low": "low_effort"},
			},
			wantChat: map[string]interface{}{"enable_thinking": true, "low_effort": true},
		},
		{
			name:      "disabled activation",
			reasoning: &config.ExternalModelReasoningConfig{Family: "test", UseReasoning: &disabled},
			family: config.ReasoningFamilyConfig{
				Type: config.ReasoningFamilyTypeReasoningEffort, Parameter: "reasoning_effort", ActivationParameter: "enable_thinking",
				Levels: []string{"low", "high"}, Default: "high",
			},
			wantChat: map[string]interface{}{"enable_thinking": false},
		},
		{
			name:      "disabled effort sentinel",
			reasoning: &config.ExternalModelReasoningConfig{Family: "test", UseReasoning: &disabled},
			family: config.ReasoningFamilyConfig{
				Type: config.ReasoningFamilyTypeReasoningEffort, Parameter: "reasoning_effort",
				Levels: []string{"low", "high"}, Default: "high", Disabled: "no_think",
			},
			wantChat: map[string]interface{}{"reasoning_effort": "no_think"},
		},
		{
			name:      "enabled reasoning mode",
			reasoning: &config.ExternalModelReasoningConfig{Family: "test", UseReasoning: &enabled},
			family: config.ReasoningFamilyConfig{
				Type: config.ReasoningFamilyTypeReasoningMode, Parameter: "thinking_mode",
				Modes: []string{config.ReasoningModeAdaptive, config.ReasoningModeDisabled}, DefaultMode: config.ReasoningModeDisabled,
			},
			wantChat: map[string]interface{}{"thinking_mode": "adaptive"},
		},
		{
			name:      "disabled reasoning mode",
			reasoning: &config.ExternalModelReasoningConfig{Family: "test", UseReasoning: &disabled},
			family: config.ReasoningFamilyConfig{
				Type: config.ReasoningFamilyTypeReasoningMode, Parameter: "thinking_mode",
				Modes: []string{config.ReasoningModeEnabled, config.ReasoningModeDisabled}, DefaultMode: config.ReasoningModeEnabled,
			},
			wantChat: map[string]interface{}{"thinking_mode": "disabled"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var families map[string]config.ReasoningFamilyConfig
			if tt.reasoning != nil {
				families = map[string]config.ReasoningFamilyConfig{"test": tt.family}
			}
			control, err := resolveExternalModelReasoningControl(families, &config.ExternalModelConfig{Reasoning: tt.reasoning})
			if err != nil {
				t.Fatalf("resolveExternalModelReasoningControl() error = %v", err)
			}
			requestToEncode := vllmChatCompletionRequest{
				Model:     "judge",
				ExtraBody: map[string]interface{}{"guided_json": map[string]interface{}{"type": "object"}},
			}
			if control != nil {
				requestToEncode.ReasoningEffort = control.reasoningEffort
				requestToEncode.ChatTemplateKwargs = control.chatTemplateKwargs
			}
			encoded, err := json.Marshal(requestToEncode)
			if err != nil {
				t.Fatalf("Marshal() error = %v", err)
			}
			var request map[string]interface{}
			if err := json.Unmarshal(encoded, &request); err != nil {
				t.Fatalf("Unmarshal() error = %v", err)
			}
			if _, leaked := request["extra_body"].(map[string]interface{})["reasoning_effort"]; leaked {
				t.Fatalf("reasoning control leaked into extra_body: %s", encoded)
			}
			if tt.wantEffort == "" {
				if _, exists := request["reasoning_effort"]; exists {
					t.Fatalf("unexpected reasoning_effort: %s", encoded)
				}
			} else if request["reasoning_effort"] != tt.wantEffort {
				t.Errorf("reasoning_effort = %v, want %q", request["reasoning_effort"], tt.wantEffort)
			}
			if tt.wantChat == nil {
				if _, exists := request["chat_template_kwargs"]; exists {
					t.Fatalf("unexpected chat_template_kwargs: %s", encoded)
				}
			} else {
				got, ok := request["chat_template_kwargs"].(map[string]interface{})
				if !ok {
					t.Fatalf("chat_template_kwargs = %T, want object: %s", request["chat_template_kwargs"], encoded)
				}
				if !mapsEqual(got, tt.wantChat) {
					t.Errorf("chat_template_kwargs = %v, want %v", got, tt.wantChat)
				}
			}
		})
	}
}

func mapsEqual(got, want map[string]interface{}) bool {
	if len(got) != len(want) {
		return false
	}
	for key, wantValue := range want {
		if got[key] != wantValue {
			return false
		}
	}
	return true
}
