package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestNewVLLMJailbreakInference tests creating vLLM inference with external config
func TestNewVLLMJailbreakInference(t *testing.T) {
	tests := []struct {
		name             string
		externalCfg      *config.ExternalModelConfig
		defaultThreshold float32
		expectedThresh   float32
		expectedParser   string
		expectedTimeout  int
		expectError      bool
	}{
		{
			name: "full config with all fields",
			externalCfg: &config.ExternalModelConfig{
				ModelEndpoint: config.ClassifierVLLMEndpoint{
					Address: "192.168.1.100",
					Port:    8080,
				},
				ModelName:      "qwen_guard",
				Threshold:      0.8,
				ParserType:     "qwen3guard",
				TimeoutSeconds: 60,
			},
			defaultThreshold: 0.7,
			expectedThresh:   0.8,          // From external config
			expectedParser:   "qwen3guard", // From external config
			expectedTimeout:  60,           // From external config
			expectError:      false,
		},
		{
			name: "config without threshold - use default",
			externalCfg: &config.ExternalModelConfig{
				ModelEndpoint: config.ClassifierVLLMEndpoint{
					Address: "192.168.1.100",
					Port:    8080,
				},
				ModelName:  "qwen_guard",
				ParserType: "qwen3guard",
				// Threshold not set - should use default
			},
			defaultThreshold: 0.7,
			expectedThresh:   0.7,          // From default
			expectedParser:   "qwen3guard", // From external config
			expectedTimeout:  30,           // Default timeout
			expectError:      false,
		},
		{
			name: "config without parser type - use auto",
			externalCfg: &config.ExternalModelConfig{
				ModelEndpoint: config.ClassifierVLLMEndpoint{
					Address: "192.168.1.100",
					Port:    8080,
				},
				ModelName: "qwen_guard",
				// ParserType not set - should use "auto"
			},
			defaultThreshold: 0.7,
			expectedThresh:   0.7,
			expectedParser:   "auto", // Default parser
			expectedTimeout:  30,     // Default timeout
			expectError:      false,
		},
		{
			name: "missing endpoint address - should error",
			externalCfg: &config.ExternalModelConfig{
				ModelName: "qwen_guard",
			},
			defaultThreshold: 0.7,
			expectError:      true,
		},
		{
			name: "missing model name - should error",
			externalCfg: &config.ExternalModelConfig{
				ModelEndpoint: config.ClassifierVLLMEndpoint{
					Address: "192.168.1.100",
					Port:    8080,
				},
			},
			defaultThreshold: 0.7,
			expectError:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inference, err := NewVLLMJailbreakInference(tt.externalCfg, tt.defaultThreshold)

			if tt.expectError {
				if err == nil {
					t.Errorf("expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			if inference.threshold != tt.expectedThresh {
				t.Errorf("expected threshold %f, got %f", tt.expectedThresh, inference.threshold)
			}
			if inference.parserType != tt.expectedParser {
				t.Errorf("expected parser %s, got %s", tt.expectedParser, inference.parserType)
			}
			expectedTimeoutSec := tt.expectedTimeout
			actualTimeoutSec := int(inference.timeout.Seconds())
			if actualTimeoutSec != expectedTimeoutSec {
				t.Errorf("expected timeout %d seconds, got %d seconds", expectedTimeoutSec, actualTimeoutSec)
			}
		})
	}
}

// TestFindExternalModelByRole tests the external model lookup
func TestFindExternalModelByRole(t *testing.T) {
	cfg := &config.RouterConfig{
		ExternalModels: []config.ExternalModelConfig{
			{
				Provider:  "vllm",
				ModelRole: config.ModelRoleClassification,
				ModelName: "classifier_model",
			},
			{
				Provider:  "vllm",
				ModelRole: config.ModelRoleGuardrail,
				ModelName: "guard_model",
			},
			{
				Provider:  "openai",
				ModelRole: config.ModelRoleScoring,
				ModelName: "scorer_model",
			},
		},
	}

	tests := []struct {
		name          string
		role          string
		expectedModel string
		shouldFind    bool
	}{
		{
			name:          "find guardrail model",
			role:          config.ModelRoleGuardrail,
			expectedModel: "guard_model",
			shouldFind:    true,
		},
		{
			name:          "find classification model",
			role:          config.ModelRoleClassification,
			expectedModel: "classifier_model",
			shouldFind:    true,
		},
		{
			name:       "role not found",
			role:       "nonexistent",
			shouldFind: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := cfg.FindExternalModelByRole(tt.role)
			if tt.shouldFind {
				if result == nil {
					t.Errorf("expected to find model with role %s, but got nil", tt.role)
				} else if result.ModelName != tt.expectedModel {
					t.Errorf("expected model name %s, got %s", tt.expectedModel, result.ModelName)
				}
			} else {
				if result != nil {
					t.Errorf("expected not to find model with role %s, but got %v", tt.role, result)
				}
			}
		})
	}
}
