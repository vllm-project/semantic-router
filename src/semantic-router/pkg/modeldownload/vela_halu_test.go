package modeldownload

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestVelaHaluDefaultIsServedByTheRuntime(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(`version: v0.3
listeners: []
providers:
  defaults: {model: backend}
  models:
    - name: backend
      backend_refs: [{name: local, provider: vllm, endpoint: "127.0.0.1:8000"}]
routing:
  decisions:
    - name: grounded
      rules: {operator: AND, conditions: []}
      modelRefs: [{model: backend}]
      plugins:
        - type: hallucination
          configuration: {enabled: true}
global:
  model_catalog:
    modules:
      hallucination_mitigation:
        enabled: true
        fact_check: {model_ref: "", model_id: ""}
`))
	if err != nil {
		t.Fatal(err)
	}
	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(specs) != 0 {
		t.Fatalf("router downloads models for a runtime-served detector: %+v", specs)
	}
	assertRuntimeServed(t, cfg, specs, config.DefaultSystemModels().HallucinationDetector)
}
