package config

import (
	"fmt"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

func reasoningMinOutputTokensYAML(threshold string) []byte {
	return []byte(fmt.Sprintf(`
version: v0.5
providers:
  defaults:
    model: worker
%s  models:
    - name: worker
      reasoning:
        family: qwen3
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
    - name: always-on
      reasoning:
        family: qwen3-always-on
      backend_refs:
        - endpoint: 127.0.0.1:8001
          provider: vllm
recipes:
  - name: default
`, threshold))
}

func TestReasoningMinOutputTokensRoundTrips(t *testing.T) {
	cfg, err := ParseYAMLBytes(reasoningMinOutputTokensYAML("    reasoning_min_output_tokens: 512\n"))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	if cfg.ReasoningMinOutputTokens != 512 {
		t.Fatalf("runtime reasoning_min_output_tokens = %d, want 512", cfg.ReasoningMinOutputTokens)
	}
	exported := CanonicalConfigFromRouterConfig(cfg)
	if got := exported.Providers.Defaults.ReasoningMinOutputTokens; got != 512 {
		t.Fatalf("exported reasoning_min_output_tokens = %d, want 512", got)
	}
	encoded, err := yaml.Marshal(exported)
	if err != nil {
		t.Fatalf("yaml.Marshal() error = %v", err)
	}
	replayed, err := ParseYAMLBytes(encoded)
	if err != nil {
		t.Fatalf("reparse exported config: %v\n%s", err, encoded)
	}
	if replayed.ReasoningMinOutputTokens != 512 {
		t.Fatalf("replayed reasoning_min_output_tokens = %d, want 512", replayed.ReasoningMinOutputTokens)
	}

	unset, err := ParseYAMLBytes(reasoningMinOutputTokensYAML(""))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() without the threshold error = %v", err)
	}
	encoded, err = yaml.Marshal(CanonicalConfigFromRouterConfig(unset))
	if err != nil {
		t.Fatalf("yaml.Marshal() error = %v", err)
	}
	if unset.ReasoningMinOutputTokens != 0 || strings.Contains(string(encoded), "reasoning_min_output_tokens") {
		t.Fatalf("an unset threshold became %d and exported as:\n%s", unset.ReasoningMinOutputTokens, encoded)
	}
}

func TestReasoningMinOutputTokensMustNotBeNegative(t *testing.T) {
	_, err := ParseYAMLBytes(reasoningMinOutputTokensYAML("    reasoning_min_output_tokens: -1\n"))
	if err == nil || !strings.Contains(err.Error(), "providers.defaults.reasoning_min_output_tokens must not be negative") {
		t.Fatalf("ParseYAMLBytes() error = %v, want the negative threshold refused", err)
	}
}

// A model can run with its reasoning off only when its family has an off
// control its Provider bindings can send, as a use_reasoning: false lane needs.
func TestCanProjectReasoningOffFollowsTheModelsFamily(t *testing.T) {
	cfg, err := ParseYAMLBytes(reasoningMinOutputTokensYAML(""))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	for model, want := range map[string]bool{"worker": true, "always-on": false, "unknown": false} {
		if got := cfg.CanProjectReasoningOff(model); got != want {
			t.Errorf("CanProjectReasoningOff(%q) = %t, want %t", model, got, want)
		}
	}
}
