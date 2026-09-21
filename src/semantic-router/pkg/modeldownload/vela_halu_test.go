package modeldownload

import (
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestVelaHaluPublishedNativeInventory(t *testing.T) {
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
          configuration: {enabled: true, use_nli: false}
global:
  model_catalog:
    modules:
      hallucination_mitigation:
        enabled: true
        fact_check: {model_ref: "", model_id: ""}
        explainer: {model_ref: "", model_id: ""}
`))
	if err != nil {
		t.Fatal(err)
	}
	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	registered := config.GetModelByPath("models/Vela-1.0-Encoder-307M-Halu")
	var spec *ModelSpec
	for index := range specs {
		if specs[index].LocalPath == registered.LocalPath {
			spec = &specs[index]
			break
		}
	}
	if spec == nil {
		t.Fatalf("profile did not provision Halu: %+v", specs)
	}
	if spec.Revision != registered.Revision || spec.RepoID != registered.RepoID || spec.CheckONNX {
		t.Fatalf("incorrect public artifact: %+v", spec)
	}
	for _, file := range []string{"config.json", "tokenizer.json", "operating_point.json"} {
		if !slices.Contains(spec.RequiredFiles, file) {
			t.Errorf("published Halu contract missing %s: %+v", file, spec)
		}
	}
	if len(spec.RequiredFileGroups) != 1 || !slices.Contains(spec.RequiredFileGroups[0], "*.safetensors") {
		t.Fatalf("default does not require native weights: %+v", spec)
	}
	for _, candidate := range specs {
		if candidate.LocalPath == "models/mom-halugate-detector" || candidate.LocalPath == "models/mom-halugate-explainer" {
			t.Fatalf("profile retained old detector or unused NLI: %+v", candidate)
		}
	}
}
