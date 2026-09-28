package config

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

const canonicalReasoningActivationYAML = `
version: v0.3
providers:
  models:
    - name: private-model
      api_format: openai
      reasoning:
        type: reasoning_effort
        parameter: reasoning_effort
        activation_parameter: enable_thinking
        levels: [disabled, low, high]
        default: low
        disabled: disabled
        modes: [enabled, disabled]
        default_mode: enabled
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
routing: {}
`

func TestCanonicalInlineReasoningActivationParameterRoundTrips(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(canonicalReasoningActivationYAML))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}

	const familyID = "operator/private-model-reasoning"
	if got := cfg.ReasoningFamilies[familyID].ActivationParameter; got != "enable_thinking" {
		t.Fatalf("effective activation_parameter = %q, want enable_thinking", got)
	}

	exported := CanonicalConfigFromRouterConfig(cfg)
	reasoning := exported.Providers.Models[0].Reasoning
	if reasoning == nil || reasoning.ActivationParameter != "enable_thinking" {
		t.Fatalf("exported inline reasoning = %#v", reasoning)
	}
	encoded, err := yaml.Marshal(exported)
	if err != nil {
		t.Fatalf("yaml.Marshal() error = %v", err)
	}
	replayed, err := ParseYAMLBytes(encoded)
	if err != nil {
		t.Fatalf("reparse exported config: %v\n%s", err, encoded)
	}
	if got := replayed.ReasoningFamilies[familyID].ActivationParameter; got != "enable_thinking" {
		t.Fatalf("replayed activation_parameter = %q, want enable_thinking", got)
	}
}

func TestCanonicalInlineReasoningActivationParameterIsValidated(t *testing.T) {
	invalid := strings.Replace(
		canonicalReasoningActivationYAML,
		"activation_parameter: enable_thinking",
		"activation_parameter: reasoning_effort",
		1,
	)
	_, err := ParseYAMLBytes([]byte(invalid))
	if err == nil || !strings.Contains(err.Error(), "activation_parameter must differ from parameter") {
		t.Fatalf("ParseYAMLBytes() error = %v, want activation parameter validation", err)
	}
}
