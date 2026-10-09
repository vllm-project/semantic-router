package modeldownload

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const velaShieldPath = "models/Vela-1.0-Encoder-307M-Shield"

const velaShieldBaseYAML = `version: v0.3
listeners: []
providers:
  defaults: {model: backend}
  models:
    - name: backend
      backend_refs: [{name: local, provider: vllm, endpoint: "127.0.0.1:8000"}]
`

const velaShieldGlobalYAML = velaShieldBaseYAML + `routing:
  signals:
    safety:
      - {name: unsafe-content, threshold: 0.5}
  decisions:
    - name: unsafe
      rules: {operator: AND, conditions: [{type: safety, name: unsafe-content}]}
      modelRefs: [{model: backend}]
global:
  model_catalog:
    modules:
      safety:
        safety:
          model_id: models/Vela-1.0-Encoder-307M-Shield
`

const velaShieldRecipeYAML = velaShieldBaseYAML + `entrypoints:
  - model_names: [care]
    recipe: care
recipes:
  - name: care
    routing:
      model_bindings:
        safety.unsafe-content: {deployment: shield, adapter: modernbert, contract: label_distribution.v1}
      signals:
        safety:
          - {name: unsafe-content, threshold: 0.5}
      decisions:
        - name: unsafe
          rules: {operator: AND, conditions: [{type: safety, name: unsafe-content}]}
          modelRefs: [{model: backend}]
global:
  model_catalog:
    deployments:
      shield:
        artifact: models/Vela-1.0-Encoder-307M-Shield
        revision: a981a99eeb05a2859b88b5cee9af4352897ec4ec
        provider: model_runtime
`

func TestVelaShieldSelectionIsServedByTheRuntime(t *testing.T) {
	for name, raw := range map[string]string{"global": velaShieldGlobalYAML, "recipe": velaShieldRecipeYAML} {
		t.Run(name, func(t *testing.T) {
			cfg, err := config.ParseYAMLBytes([]byte(raw))
			if err != nil {
				t.Fatal(err)
			}
			specs, err := BuildModelSpecs(cfg)
			if err != nil {
				t.Fatal(err)
			}
			if len(specs) != 0 {
				t.Fatalf("router downloads models for a runtime-served selection: %+v", specs)
			}
			assertRuntimeServed(t, cfg, specs, velaShieldPath)
			// The selected model replaces Vela Safety instead of adding to it.
			for _, deployment := range config.ModelRuntimeDeploymentsInUse(cfg) {
				if strings.Contains(deployment.Artifact, "Vela-1.0-Encoder-307M-Safety") {
					t.Fatalf("Shield selection also serves Vela Safety: %+v", deployment)
				}
			}
		})
	}
}
