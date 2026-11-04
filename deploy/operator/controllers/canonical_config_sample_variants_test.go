package controllers

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"gopkg.in/yaml.v3"
	kubeyaml "sigs.k8s.io/yaml"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// These configuration families had dedicated deployment workers whose common
// unconditional route did not observe their distinguishing configuration. Test
// the actual CR -> canonical YAML -> router loader boundary without a cluster.
func TestMMBertSampleSurvivesRouterConfigRoundTrip(t *testing.T) {
	cfg := sampleRouterConfig(t, "vllm.ai_v1alpha1_semanticrouter_mmbert.yaml")
	if cfg.MmBertModelPath != "models/Vela-1.0-Encoder-307M-Embedding" || !cfg.UseCPU {
		t.Fatalf("sample embedding artifact/device changed: %+v", cfg.EmbeddingModels)
	}
	embedding := cfg.EmbeddingConfig
	if embedding.ModelType != "mmbert" || embedding.TargetLayer != 6 || embedding.TargetDimension != 256 ||
		!embedding.PreloadEmbeddings || embedding.EnableSoftMatching == nil || !*embedding.EnableSoftMatching ||
		embedding.MinScoreThreshold != .5 {
		t.Fatalf("sample embedding settings lost in translation: %+v", embedding)
	}
	cache := cfg.SemanticCache
	if !cache.Enabled || cache.BackendType != "memory" || cache.EmbeddingModel != "mmbert" ||
		cache.MaxEntries != 5000 || cache.TTLSeconds != 7200 || cache.EvictionPolicy != "lru" ||
		cache.SimilarityThreshold == nil || *cache.SimilarityThreshold != .85 {
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
		original := sample.Spec.Config.ComplexityRules[i]
		if rule.Name != wantNames[i] || rule.Threshold != .3 || !rule.ThresholdSet ||
			len(rule.Hard.Candidates) != 5 || len(rule.Easy.Candidates) != 5 ||
			!reflect.DeepEqual(rule.Hard.Candidates, original.Hard.Candidates) ||
			!reflect.DeepEqual(rule.Easy.Candidates, original.Easy.Candidates) {
			t.Fatalf("sample complexity rule lost names, thresholds or prototypes: %+v", rule)
		}
		if i < 2 && rule.Composer != nil {
			t.Fatalf("unconditional sample rule gained a composer: %+v", rule)
		}
	}
	composer := cfg.ComplexityRules[2].Composer
	if composer == nil || composer.Operator != "AND" || len(composer.Conditions) != 1 ||
		composer.Conditions[0].Type != "domain" || composer.Conditions[0].Name != "medical" {
		t.Fatalf("medical sample domain condition lost in translation: %+v", composer)
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
