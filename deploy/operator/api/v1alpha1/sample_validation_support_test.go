package v1alpha1

import "testing"

const (
	sampleMMBert      = "vllm.ai_v1alpha1_semanticrouter_mmbert.yaml"
	sampleComplexity  = "vllm.ai_v1alpha1_semanticrouter_complexity.yaml"
	sampleSimple      = "vllm.ai_v1alpha1_semanticrouter_simple.yaml"
	sampleRedisCache  = "vllm.ai_v1alpha1_semanticrouter_redis_cache.yaml"
	sampleMilvusCache = "vllm.ai_v1alpha1_semanticrouter_milvus_cache.yaml"
)

type sampleContractAssertion func(*testing.T, *SemanticRouter)

var sampleContractAssertions = map[string]sampleContractAssertion{
	sampleMMBert:      assertMMBertSampleContract,
	sampleComplexity:  assertComplexitySampleContract,
	sampleRedisCache:  assertRedisCacheSampleContract,
	sampleMilvusCache: assertMilvusCacheSampleContract,
}

func assertSampleSpecificContract(t *testing.T, filename string, sr *SemanticRouter) {
	t.Helper()
	if assertFn, ok := sampleContractAssertions[filename]; ok {
		assertFn(t, sr)
	}
}

func assertMMBertSampleContract(t *testing.T, sr *SemanticRouter) {
	t.Helper()
	if sr.Spec.Config.EmbeddingModels == nil {
		t.Error("mmbert sample should have embedding_models configured")
		return
	}
	if sr.Spec.Config.EmbeddingModels.MmBertModelPath == "" {
		t.Error("mmbert sample should have mmbert_model_path set")
	}
	if sr.Spec.Config.EmbeddingModels.EmbeddingConfig == nil {
		t.Error("mmbert sample should have embedding_config")
		return
	}
	if sr.Spec.Config.EmbeddingModels.EmbeddingConfig.ModelType != "mmbert" {
		t.Errorf(
			"mmbert sample embedding_config model_type = %v, want mmbert",
			sr.Spec.Config.EmbeddingModels.EmbeddingConfig.ModelType,
		)
	}
	validLayers := map[int]bool{3: true, 6: true, 11: true, 22: true}
	if !validLayers[sr.Spec.Config.EmbeddingModels.EmbeddingConfig.TargetLayer] {
		t.Errorf(
			"mmbert sample target_layer = %v, want one of 3, 6, 11, 22",
			sr.Spec.Config.EmbeddingModels.EmbeddingConfig.TargetLayer,
		)
	}
	if sr.Spec.Config.SemanticCache != nil && sr.Spec.Config.SemanticCache.EmbeddingModel != "mmbert" {
		t.Errorf(
			"mmbert sample should use embedding_model: mmbert, got %v",
			sr.Spec.Config.SemanticCache.EmbeddingModel,
		)
	}
}

func assertComplexitySampleContract(t *testing.T, sr *SemanticRouter) {
	t.Helper()
	if len(sr.Spec.Config.ComplexityRules) == 0 {
		t.Error("complexity sample should have complexity_rules configured")
	}
	for _, rule := range sr.Spec.Config.ComplexityRules {
		if rule.Name == "" {
			t.Error("complexity rule should have a name")
		}
		if len(rule.Hard.Candidates) == 0 {
			t.Errorf("complexity rule %s should have hard candidates", rule.Name)
		}
		if len(rule.Easy.Candidates) == 0 {
			t.Errorf("complexity rule %s should have easy candidates", rule.Name)
		}
	}
}

func assertRedisCacheSampleContract(t *testing.T, sr *SemanticRouter) {
	t.Helper()
	if sr.Spec.Config.EmbeddingModels == nil {
		t.Error("redis cache sample should have embedding_models configured")
	}
	if sr.Spec.Config.SemanticCache == nil {
		return
	}
	if sr.Spec.Config.SemanticCache.BackendType != "redis" {
		t.Errorf("redis cache sample backend_type = %v, want redis", sr.Spec.Config.SemanticCache.BackendType)
	}
	if sr.Spec.Config.SemanticCache.EmbeddingModel != "" && sr.Spec.Config.SemanticCache.EmbeddingModel == "bert" {
		t.Logf("redis cache sample could showcase new embedding models (qwen3/gemma)")
	}
}

func assertMilvusCacheSampleContract(t *testing.T, sr *SemanticRouter) {
	t.Helper()
	if sr.Spec.Config.EmbeddingModels == nil {
		t.Error("milvus cache sample should have embedding_models configured")
	}
	if sr.Spec.Config.SemanticCache != nil && sr.Spec.Config.SemanticCache.BackendType != "milvus" {
		t.Errorf("milvus cache sample backend_type = %v, want milvus", sr.Spec.Config.SemanticCache.BackendType)
	}
}
