package modeldownload

import (
	"slices"
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
        provider: candle
`

func TestVelaShieldSelectionDownloadsOnlyThePinnedRootClassifier(t *testing.T) {
	registered := config.GetModelByPath(velaShieldPath)
	if registered == nil {
		t.Fatal("Shield is not registered")
	}
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
			// The selected model replaces Vela Safety instead of adding to it.
			if len(specs) != 1 || specs[0].LocalPath != velaShieldPath {
				t.Fatalf("expected only the Shield download, got %+v", specs)
			}
			spec := specs[0]
			if spec.RepoID != registered.RepoID || spec.Revision != registered.Revision {
				t.Fatalf("Shield download lost its pinned identity: %+v", spec)
			}
			for _, file := range []string{
				"lc/model.safetensors", "lc/head.safetensors", "lc/load_lc.py",
				"heads/avg/extra_heads.safetensors", "heads/seed42/extra_heads.safetensors",
				"heads/seed43/extra_heads.safetensors", "heads/seed44/extra_heads.safetensors",
				"heads/load_heads.py", "demo.py", "DEMO_OUTPUT.txt",
			} {
				if !revisionArtifactExcluded(file, spec.ExcludePatterns) {
					t.Errorf("unused Shield artifact would be downloaded: %s", file)
				}
			}
			for _, file := range []string{"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "label_mapping.json"} {
				if revisionArtifactExcluded(file, spec.ExcludePatterns) {
					t.Errorf("root classifier file was excluded: %s", file)
				}
			}
			args := buildDownloadArgs(spec)
			for _, pattern := range []string{"lc/*", "heads/*"} {
				if !slices.Contains(args, pattern) {
					t.Errorf("HF download did not receive exclusion %s: %v", pattern, args)
				}
			}
		})
	}
}
