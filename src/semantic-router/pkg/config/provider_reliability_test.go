package config

import (
	"strings"
	"testing"
)

func TestModelExecutionRoundTripsCanonicalConfig(t *testing.T) {
	document := strings.Replace(entrypointRulesYAML, "  - name: model-a\n", `  - name: model-a
    runtime:
      max_retries: 2
      request_timeout: 45s
      stream_timeout: 3m
`, 1)
	parsed, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatalf("ParseYAMLBytes: %v", err)
	}
	execution := parsed.ModelConfig["model-a"].Execution
	if execution.MaxRetries != 2 || execution.RequestTimeout != "45s" || execution.StreamTimeout != "3m" {
		t.Fatalf("execution did not normalize: %#v", execution)
	}
	exported := CanonicalConfigFromRouterConfig(parsed)
	var exportedExecution ModelExecutionSettings
	for _, model := range exported.Models {
		if model.Name == "model-a" {
			exportedExecution = model.Execution
		}
	}
	if len(exported.Models) != 3 || exportedExecution != execution {
		t.Fatalf("execution did not round trip: %+v", exported.Models)
	}
}

func TestProviderReliabilityRejectsUnsafeValues(t *testing.T) {
	err := validateProviderReliability("model-a", ProviderReliability{
		LBPolicy:   "random",
		RetryCount: 8,
	})
	if err == nil {
		t.Fatal("invalid reliability config must be rejected")
	}
}
