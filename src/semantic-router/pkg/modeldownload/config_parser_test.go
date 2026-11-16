package modeldownload

import (
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// The balance recipe's classifiers are runtime-served; the router downloads
// only the embedding it runs itself.
// The balance recipe routes on decision-model questions; the model runtime
// serves that model, so the router downloads nothing for it.
var expectedAMDModelSpecs = []string{}

func TestIsModelDirectory(t *testing.T) {
	tests := []struct {
		path     string
		expected bool
	}{
		{"models/bert-base-uncased", true},
		{"models/Vela-1.0-Encoder-307M-Safety", true},
		{"models/Vela-1.0-Encoder-307M-Safety/model.safetensors", false},
		{"models/gmtrouter.pt", false},
		{"models/lora_model/adapter_config.json", false},
		{"models/mapping.json", false},
		{"config/tools_db.json", false},
		{"models/mom-embedding-pro", true},
		{"models/mom-embedding-flash", true},
	}

	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			result := isModelDirectory(tt.path)
			if result != tt.expected {
				t.Errorf("isModelDirectory(%s) = %v, expected %v", tt.path, result, tt.expected)
			}
		})
	}
}

func TestBuildModelSpecsSkipsUnusedRouterOwnedDefaultsForScratchCanonicalConfig(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(`
version: v0.5
listeners:
  - name: http-8888
    address: 0.0.0.0
    port: 8888
providers:
  defaults:
    model: openai/gpt-oss-120b
  models:
    - name: openai/gpt-oss-120b
      provider_model_id: openai/gpt-oss-120b
      backend_refs:
        - name: primary
          provider: vllm
          endpoint: localhost:8000
          protocol: http
          weight: 100
recipes:
  - name: default
    decisions:
      - name: default-route
        priority: 100
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: openai/gpt-oss-120b
            use_reasoning: false
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}

	if len(specs) != 0 {
		t.Fatalf("unused defaults requested model downloads: %+v", specs)
	}
}

func TestBuildModelSpecsSkipsUnusedRouterOwnedDefaultsForSparseAMDGlobalOverride(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(`
version: v0.5
listeners:
  - name: http-8888
    address: 0.0.0.0
    port: 8888
providers:
  defaults:
    model: openai/gpt-oss-120b
  models:
    - name: openai/gpt-oss-120b
      provider_model_id: openai/gpt-oss-120b
      backend_refs:
        - name: primary
          provider: vllm
          endpoint: localhost:8000
          protocol: http
          weight: 100
recipes:
  - name: default
    decisions:
      - name: default-route
        priority: 100
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: openai/gpt-oss-120b
            use_reasoning: false
global:
  model_catalog:
    deployments:
      vela-embedding:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Embedding
        device: rocm:0
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}

	if len(specs) != 0 {
		t.Fatalf("unused defaults requested model downloads: %+v", specs)
	}
}

func TestBuildModelSpecsAcceptsReferenceConfig(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("failed to resolve reference config path")
	}

	configPath := filepath.Clean(filepath.Join(filepath.Dir(file), "../../../../config/config.yaml"))
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read %s: %v", configPath, err)
	}

	cfg, err := config.ParseYAMLBytes(data)
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}

	// embeddings.default names vela-embedding, so the model runtime serves it.
	assertRuntimeServed(t, cfg, specs, "models/Vela-1.0-Encoder-307M-Embedding")
	served := cfg.RuntimeServedModelPaths()
	for _, spec := range specs {
		if served[spec.LocalPath] && !spec.FilesOnly {
			t.Fatalf("router downloads the runtime-served %s: %+v", spec.LocalPath, spec)
		}
	}
	assertRuntimeServed(t, cfg, specs, config.Vela3SignalModel)
}

func TestBuildModelSpecsIncludesAllAMDDeployModels(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("failed to resolve amd config path")
	}

	configPath := filepath.Clean(filepath.Join(filepath.Dir(file), "../../../../config/recipes/balance/config.yaml"))
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read %s: %v", configPath, err)
	}

	cfg, err := config.ParseYAMLBytes(data)
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}

	assertContainsAllModelSpecs(t, specs,
		expectedAMDModelSpecs...,
	)
	if len(specs) != len(expectedAMDModelSpecs) {
		t.Fatalf("BuildModelSpecs() returned %d specs, want %d", len(specs), len(expectedAMDModelSpecs))
	}
}

func TestBuildModelSpecsSkipsRouterOwnedDefaultsForAgentSmokeConfigs(t *testing.T) {
	for _, relParts := range [][]string{
		{"..", "..", "..", "..", "e2e", "config", "config.agent-smoke.cpu.yaml"},
		{"..", "..", "..", "..", "e2e", "config", "config.agent-smoke.amd.yaml"},
	} {
		rel := filepath.Join(relParts...)
		t.Run(rel, func(t *testing.T) {
			_, file, _, ok := runtime.Caller(0)
			if !ok {
				t.Fatal("failed to resolve smoke config path")
			}

			configPath := filepath.Clean(filepath.Join(filepath.Dir(file), rel))
			data, err := os.ReadFile(configPath)
			if err != nil {
				t.Fatalf("read %s: %v", configPath, err)
			}

			cfg, err := config.ParseYAMLBytes(data)
			if err != nil {
				t.Fatalf("ParseYAMLBytes() error = %v", err)
			}

			specs, err := BuildModelSpecs(cfg)
			if err != nil {
				t.Fatalf("BuildModelSpecs() error = %v", err)
			}
			if len(specs) != 0 {
				t.Fatalf("BuildModelSpecs() returned %d specs, want 0: %#v", len(specs), specs)
			}
		})
	}
}

func TestBuildModelSpecsLeavesTheMemoryEmbeddingToTheRuntimeForMemoryE2EConfigs(t *testing.T) {
	for _, relParts := range [][]string{
		{"..", "..", "..", "..", "e2e", "config", "config.memory-user.yaml"},
		{"..", "..", "..", "..", "e2e", "config", "config.memory-user-valkey.yaml"},
	} {
		rel := filepath.Join(relParts...)
		t.Run(rel, func(t *testing.T) {
			_, file, _, ok := runtime.Caller(0)
			if !ok {
				t.Fatal("failed to resolve memory config path")
			}

			configPath := filepath.Clean(filepath.Join(filepath.Dir(file), rel))
			data, err := os.ReadFile(configPath)
			if err != nil {
				t.Fatalf("read %s: %v", configPath, err)
			}

			cfg, err := config.ParseYAMLBytes(data)
			if err != nil {
				t.Fatalf("ParseYAMLBytes() error = %v", err)
			}

			specs, err := BuildModelSpecs(cfg)
			if err != nil {
				t.Fatalf("BuildModelSpecs() error = %v", err)
			}
			if len(specs) != 0 {
				t.Fatalf("BuildModelSpecs() returned %d specs, want 0: %#v", len(specs), specs)
			}
			assertRuntimeServed(t, cfg, specs, "models/Vela-1.0-Encoder-307M-Embedding")
		})
	}
}

func assertContainsAllModelSpecs(t *testing.T, specs []ModelSpec, wantPaths ...string) {
	t.Helper()

	gotPaths := make([]string, 0, len(specs))
	for _, spec := range specs {
		gotPaths = append(gotPaths, spec.LocalPath)
	}

	for _, want := range wantPaths {
		if !slices.Contains(gotPaths, want) {
			t.Fatalf("BuildModelSpecs() missing %q; got %v", want, gotPaths)
		}
	}
}
