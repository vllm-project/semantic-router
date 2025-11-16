package pii

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestIsPIIEnabled_LoRAFallback tests that LoRA adapters fall back to base model's PII policy
func TestIsPIIEnabled_LoRAFallback(t *testing.T) {
	tests := []struct {
		name           string
		modelConfigs   map[string]config.ModelParams
		model          string
		expectedResult bool
		description    string
	}{
		{
			name: "LoRA adapter inherits base model PII policy (enabled)",
			modelConfigs: map[string]config.ModelParams{
				"base-model": {
					PIIPolicy: config.PIIPolicy{
						AllowByDefault: false, // PII policy enabled
					},
					LoRAs: []config.LoRAAdapter{
						{Name: "science-expert"},
						{Name: "humanities-expert"},
					},
				},
			},
			model:          "humanities-expert",
			expectedResult: true, // Should inherit base model's policy (enabled)
			description:    "LoRA adapter should inherit base model's PII policy when not explicitly configured",
		},
		{
			name: "LoRA adapter inherits base model PII policy (disabled)",
			modelConfigs: map[string]config.ModelParams{
				"base-model": {
					PIIPolicy: config.PIIPolicy{
						AllowByDefault: true, // PII policy disabled
					},
					LoRAs: []config.LoRAAdapter{
						{Name: "general-expert"},
					},
				},
			},
			model:          "general-expert",
			expectedResult: false, // Should inherit base model's policy (disabled)
			description:    "LoRA adapter should inherit base model's disabled PII policy",
		},
		{
			name: "Base model PII policy check",
			modelConfigs: map[string]config.ModelParams{
				"base-model": {
					PIIPolicy: config.PIIPolicy{
						AllowByDefault: false,
					},
				},
			},
			model:          "base-model",
			expectedResult: true,
			description:    "Base model should use its own PII policy",
		},
		{
			name: "Unknown model without LoRA mapping",
			modelConfigs: map[string]config.ModelParams{
				"base-model": {
					PIIPolicy: config.PIIPolicy{
						AllowByDefault: false,
					},
				},
			},
			model:          "unknown-model",
			expectedResult: false,
			description:    "Unknown model should return false (no policy found)",
		},
		{
			name: "LoRA adapter with explicit PII policy overrides base model",
			modelConfigs: map[string]config.ModelParams{
				"base-model": {
					PIIPolicy: config.PIIPolicy{
						AllowByDefault: false, // Base model has strict policy
					},
					LoRAs: []config.LoRAAdapter{
						{Name: "permissive-lora"},
					},
				},
				"permissive-lora": {
					PIIPolicy: config.PIIPolicy{
						AllowByDefault: true, // LoRA has permissive policy
					},
				},
			},
			model:          "permissive-lora",
			expectedResult: false, // Should use LoRA's own policy (disabled)
			description:    "LoRA adapter with explicit policy should not fall back to base model",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			checker := &PolicyChecker{
				ModelConfigs: tt.modelConfigs,
			}

			result := checker.IsPIIEnabled(tt.model)

			if result != tt.expectedResult {
				t.Errorf("%s: expected %v, got %v", tt.description, tt.expectedResult, result)
			}
		})
	}
}

// TestCheckPolicy_LoRAFallback tests that CheckPolicy falls back to base model for LoRA adapters
func TestCheckPolicy_LoRAFallback(t *testing.T) {
	tests := []struct {
		name            string
		modelConfigs    map[string]config.ModelParams
		model           string
		detectedPII     []string
		expectedAllowed bool
		expectedDenied  []string
		description     string
	}{
		{
			name: "LoRA adapter inherits base model's strict PII policy",
			modelConfigs: map[string]config.ModelParams{
				"base-model": {
					PIIPolicy: config.PIIPolicy{
						AllowByDefault: false,
						PIITypes:       []string{"GPE"}, // Only allow GPE
					},
					LoRAs: []config.LoRAAdapter{
						{Name: "science-expert"},
					},
				},
			},
			model:           "science-expert",
			detectedPII:     []string{"EMAIL_ADDRESS", "CREDIT_CARD"},
			expectedAllowed: false,
			expectedDenied:  []string{"EMAIL_ADDRESS", "CREDIT_CARD"},
			description:     "LoRA should inherit base model's strict policy and block non-allowed PII",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			checker := &PolicyChecker{
				ModelConfigs: tt.modelConfigs,
			}

			allowed, deniedPII, err := checker.CheckPolicy(tt.model, tt.detectedPII)
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}

			if allowed != tt.expectedAllowed {
				t.Errorf("%s: expected allowed=%v, got %v", tt.description, tt.expectedAllowed, allowed)
			}

			if len(deniedPII) != len(tt.expectedDenied) {
				t.Errorf("%s: expected denied PII %v, got %v", tt.description, tt.expectedDenied, deniedPII)
			}
		})
	}
}
