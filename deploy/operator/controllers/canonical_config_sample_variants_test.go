package controllers

import (
	"os"
	"path/filepath"
	"testing"

	"gopkg.in/yaml.v3"
	kubeyaml "sigs.k8s.io/yaml"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// These configuration families had dedicated deployment workers whose common
// unconditional route did not observe their distinguishing configuration. Test
// the actual CR -> canonical YAML -> router loader boundary without a cluster.
func TestEmbeddingSampleSurvivesRouterConfigRoundTrip(t *testing.T) {
	cfg := sampleRouterConfig(t, "vllm.ai_v1alpha1_semanticrouter_embedding.yaml")
	if deployment, dimension, _ := cfg.ResponseCacheEmbedding(); deployment != routerconfig.EmbeddingDefaultDeployment || dimension != 256 {
		t.Fatalf("sample cache embedding lost in translation: %s at %d", deployment, dimension)
	}
	cache := cfg.SemanticCache
	if cache.BackendType != "memory" ||
		cache.MaxEntries != 5000 || cache.TTLSeconds != 7200 || cache.EvictionPolicy != "lru" {
		t.Fatalf("sample response cache settings lost in translation: %+v", cache)
	}
	// The legacy response_cache.hnsw sample field has no router store field.
	// Neither this test nor the retired live worker qualifies that setting.
}

func TestComplexitySampleSurvivesRouterConfigRoundTrip(t *testing.T) {
	const name = "vllm.ai_v1alpha1_semanticrouter_complexity.yaml"
	cfg := sampleRouterConfig(t, name)
	data, err := os.ReadFile(filepath.Join("..", "config", "samples", name))
	if err != nil {
		t.Fatal(err)
	}
	var sample vllmv1alpha1.SemanticRouter
	if err := kubeyaml.Unmarshal(data, &sample); err != nil {
		t.Fatal(err)
	}
	wantNames := []string{"code-complexity", "reasoning-complexity", "medical-complexity"}
	if len(cfg.ComplexityRules) != len(wantNames) || len(sample.Spec.Config.ComplexityRules) != len(wantNames) {
		t.Fatalf("expected all three sample complexity rules, got %+v", cfg.ComplexityRules)
	}
	for i, rule := range cfg.ComplexityRules {
		if rule.Name != wantNames[i] || rule.Description != sample.Spec.Config.ComplexityRules[i].Description {
			t.Fatalf("sample complexity rule lost its name or description: %+v", rule)
		}
	}
}

func sampleRouterConfig(t *testing.T, name string) *routerconfig.RouterConfig {
	t.Helper()
	data, err := yaml.Marshal(sampleModelConfig(t, name))
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := routerconfig.ParseYAMLBytes(data)
	if err != nil {
		t.Fatalf("sample %s cannot be loaded by the router: %v", name, err)
	}
	return cfg
}
