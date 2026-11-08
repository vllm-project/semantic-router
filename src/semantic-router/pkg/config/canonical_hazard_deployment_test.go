package config

import (
	"testing"
)

func TestDefaultHazardOverrideReplacesWholeEntry(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`version: v0.5
global:
  model_catalog:
    deployments:
      hazard:
        provider: model_runtime
        artifact: /models/operator-hazard
`))
	if err != nil {
		t.Fatal(err)
	}
	deployment := cfg.ModelDeployments["hazard"]
	if deployment.Artifact != "/models/operator-hazard" || deployment.Revision != "" || deployment.Input.MaxTokens != 0 {
		t.Fatalf("operator deployment inherited unrelated default fields: %+v", deployment)
	}
}
