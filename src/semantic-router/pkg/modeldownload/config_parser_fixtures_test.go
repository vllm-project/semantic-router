package modeldownload

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var expectedAMDModelSpecs = []string{
	"models/mmbert-embed-32k-2d-matryoshka",
	"models/mmbert32k-intent-classifier-merged",
	"models/mmbert32k-factcheck-classifier-merged",
	"models/mmbert32k-feedback-detector-merged",
}

func TestBuildModelSpecsIncludesRouterOwnedDefaultsForScratchCanonicalConfig(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(`
version: v0.3
listeners:
  - name: http-8888
    address: 0.0.0.0
    port: 8888
providers:
  defaults:
    default_model: openai/gpt-oss-120b
  models:
    - name: openai/gpt-oss-120b
      provider_model_id: openai/gpt-oss-120b
      backend_refs:
        - name: primary
          endpoint: localhost:8000
          protocol: http
          weight: 100
routing:
  modelCards:
    - name: openai/gpt-oss-120b
      modality: text
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

	assertContainsAllModelSpecs(t, specs,
		"models/mmbert-embed-32k-2d-matryoshka",
	)
}

func TestBuildModelSpecsIncludesRouterOwnedDefaultsForSparseAMDGlobalOverride(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(`
version: v0.3
listeners:
  - name: http-8888
    address: 0.0.0.0
    port: 8888
providers:
  defaults:
    default_model: openai/gpt-oss-120b
  models:
    - name: openai/gpt-oss-120b
      provider_model_id: openai/gpt-oss-120b
      backend_refs:
        - name: primary
          endpoint: localhost:8000
          protocol: http
          weight: 100
routing:
  modelCards:
    - name: openai/gpt-oss-120b
      modality: text
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
    embeddings:
      semantic:
        use_cpu: false
    modules:
      prompt_guard:
        use_cpu: false
      classifier:
        domain:
          use_cpu: false
        pii:
          use_cpu: false
      hallucination_mitigation:
        fact_check:
          use_cpu: false
        detector:
          use_cpu: false
        explainer:
          use_cpu: false
      feedback_detector:
        use_cpu: false
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}

	assertContainsAllModelSpecs(t, specs,
		"models/mmbert-embed-32k-2d-matryoshka",
	)
}

func TestBuildModelSpecsSkipsUnusedFeedbackDetectorDefaults(t *testing.T) {
	cfg := &config.RouterConfig{
		MoMRegistry: map[string]string{
			"models/mmbert32k-feedback-detector-merged": "llm-semantic-router/mmbert32k-feedback-detector-merged",
		},
		InlineModels: config.InlineModels{
			FeedbackDetector: config.FeedbackDetectorConfig{
				Enabled: true,
				ModelID: "models/mmbert32k-feedback-detector-merged",
			},
		},
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	if len(specs) != 0 {
		t.Fatalf("BuildModelSpecs() returned %d specs, want 0", len(specs))
	}
}

func TestBuildModelSpecsAcceptsReferenceConfig(t *testing.T) {
	configPath := repoRelativeConfigPath(t, "config", "config.yaml")
	cfg := parseRouterConfigFile(t, configPath)

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}

	assertContainsAllModelSpecs(t, specs,
		"models/mom-embedding-pro",
		"models/mom-embedding-flash",
		"models/mmbert-embed-32k-2d-matryoshka",
		"models/mom-embedding-light",
		"models/mmbert32k-modality-router-merged",
	)
}

func TestBuildModelSpecsIncludesAllAMDDeployModels(t *testing.T) {
	configPath := repoRelativeConfigPath(t, "deploy", "recipes", "balance.yaml")
	cfg := parseRouterConfigFile(t, configPath)

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
		{"e2e", "config", "config.agent-smoke.cpu.yaml"},
		{"e2e", "config", "config.agent-smoke.amd.yaml"},
	} {
		t.Run(filepath.Join(relParts...), func(t *testing.T) {
			cfg := parseRouterConfigFile(t, repoRelativeConfigPath(t, relParts...))

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

func TestBuildModelSpecsSkipsRouterOwnedDefaultsForMemoryE2EConfigs(t *testing.T) {
	for _, relParts := range [][]string{
		{"e2e", "config", "config.memory-user.yaml"},
		{"e2e", "config", "config.memory-user-valkey.yaml"},
	} {
		t.Run(filepath.Join(relParts...), func(t *testing.T) {
			cfg := parseRouterConfigFile(t, repoRelativeConfigPath(t, relParts...))

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

func TestBuildModelSpecsUsesManagedBERTFallbackForBERTSemanticCache(t *testing.T) {
	cfg := &config.RouterConfig{
		MoMRegistry: config.ToLegacyRegistry(),
		SemanticCache: config.SemanticCache{
			Enabled:        true,
			EmbeddingModel: "bert",
		},
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	if len(specs) != 1 {
		t.Fatalf("BuildModelSpecs() returned %d specs, want 1", len(specs))
	}
	if specs[0].LocalPath != "models/mom-embedding-light" {
		t.Fatalf("LocalPath = %q, want managed BERT fallback", specs[0].LocalPath)
	}
	if specs[0].RepoID != "sentence-transformers/all-MiniLM-L12-v2" {
		t.Fatalf("RepoID = %q, want registry-owned MiniLM fallback", specs[0].RepoID)
	}
}

func repoRelativeConfigPath(t *testing.T, parts ...string) string {
	t.Helper()

	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("failed to resolve current test path")
	}

	relParts := append([]string{filepath.Dir(file), "..", "..", "..", ".."}, parts...)
	return filepath.Clean(filepath.Join(relParts...))
}

func parseRouterConfigFile(t *testing.T, configPath string) *config.RouterConfig {
	t.Helper()

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read %s: %v", configPath, err)
	}

	cfg, err := config.ParseYAMLBytes(data)
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	return cfg
}
