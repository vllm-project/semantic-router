package modeldownload

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// assertRuntimeServed checks that a built-in model reaches the model runtime
// at its registered release and that the router downloads none of it beyond
// companion files.
func assertRuntimeServed(t *testing.T, cfg *config.RouterConfig, specs []ModelSpec, path string) {
	t.Helper()
	model := config.GetModelByPath(path)
	if model == nil {
		t.Fatalf("%s is not a built-in model", path)
	}
	if spec, ok := findSpecByPath(specs, model.LocalPath); ok && !spec.FilesOnly {
		t.Fatalf("router downloads the runtime-served %s: %+v", path, spec)
	}
	for name, deployment := range config.ModelRuntimeDeploymentsInUse(cfg) {
		if config.SameModelRepo(deployment.Artifact, model.RepoID) || config.ResolveModelPath(deployment.Artifact) == model.LocalPath {
			// An omitted revision delegates the built-in pin to the model runtime.
			if deployment.Revision != "" && deployment.Revision != model.Revision {
				t.Fatalf("deployment %q serves %s at %q, want the release %q", name, path, deployment.Revision, model.Revision)
			}
			return
		}
	}
	t.Fatalf("no model_runtime deployment in use serves %s: %+v", path, config.ModelRuntimeDeploymentsInUse(cfg))
}

func TestDefaultPIIIsServedWithItsOwnLabels(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(`
version: v0.3
providers:
  defaults:
    model: demo
  models:
    - name: demo
      backend_refs:
        - name: primary
          endpoint: localhost:8000
          protocol: http
          weight: 1
routing:
  signals:
    pii:
      - name: sensitive-contact
        threshold: 0.9
        pii_types_allowed: []
  decisions:
    - name: protect-contact
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: pii
            name: sensitive-contact
      modelRefs:
        - model: demo
`))
	if err != nil {
		t.Fatal(err)
	}
	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(specs) != 0 || cfg.PIIModel.PIIMappingPath != "" {
		t.Fatalf("default PII needs no mapping file, its card carries the labels: mapping %q, downloads %+v", cfg.PIIModel.PIIMappingPath, specs)
	}
	assertRuntimeServed(t, cfg, specs, config.DefaultSystemModels().PIIClassifier)
}
