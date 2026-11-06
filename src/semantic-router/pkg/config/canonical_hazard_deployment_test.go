package config

import (
	"testing"
)

const defaultHazardConsumerYAML = `version: v0.3
entrypoints:
  - model_names: [care]
    recipe: care
recipes:
  - name: care
    routing:
      model_bindings:
        classifier.content-risk:
          deployment: hazard
          adapter: modernbert
          contract: label_scores.v1
          operating_point:
            path: operating_point.json
            sha256: e79a78f48bf45eb38e3f5402de3b3b18eeaa822e00b42b3640bf471276290de5
      signals:
        classifiers:
          - name: content-risk
            type: local
            labels: [violence, criminal_activity, sexual_content, child_exploitation, hate, harassment_abuse, regulated_substances, weapons, self_harm, privacy, specialized_advice, misinformation]
`

func TestDefaultHazardOverrideReplacesWholeEntry(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`version: v0.3
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
