package config

import (
	"strings"
	"testing"
)

func strictV03WithGlobal(fragment string) []byte {
	return []byte(strings.Replace(strictV03AuthoringYAML, "global:\n", "global:\n"+fragment, 1))
}

func TestParseYAMLBytesRejectsLegacyUserConfigLayout(t *testing.T) {
	legacyYAML := []byte(`
version: v0.3
signals: {}
decisions: []
providers:
  default_model: qwen2.5:3b
  models:
    - name: qwen2.5:3b
      endpoints:
        - endpoint: 127.0.0.1:11434
`)

	_, err := testAuthoringParser(t).ParseYAMLBytes(legacyYAML)
	if err == nil {
		t.Fatal("expected legacy user config layout to be rejected")
	}
	for _, fragment := range []string{"providers.default_model", "providers.models[0].endpoints"} {
		if !strings.Contains(err.Error(), fragment) {
			t.Fatalf("expected error to mention %q, got: %s", fragment, err)
		}
	}
}

func TestParseYAMLBytesRejectsTopLevelLegacyRuntimeLayout(t *testing.T) {
	_, err := testAuthoringParser(t).ParseYAMLBytes([]byte(`
version: v0.3
default_model: qwen2.5:3b
semantic_cache: {enabled: false}
`))
	if err == nil {
		t.Fatal("expected top-level legacy runtime layout to be rejected")
	}
	for _, fragment := range []string{
		"config file must use the current v0.3 version/listeners/providers/routing/recipes/entrypoints/global authoring schema",
		"unexpected top-level keys: default_model, semantic_cache",
	} {
		if !strings.Contains(err.Error(), fragment) {
			t.Fatalf("expected error to mention %q, got: %s", fragment, err)
		}
	}
}

func TestParseYAMLBytesRequiresExactV03Version(t *testing.T) {
	document := strings.Replace(strictV03AuthoringYAML, "version: v0.3", "version: ' v0.3 '", 1)
	_, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
	if err == nil || !strings.Contains(err.Error(), "version must be v0.3") {
		t.Fatalf("expected surrounding-whitespace version rejection, got: %v", err)
	}
}

func TestParseYAMLBytesRejectsUnknownGlobalModulesField(t *testing.T) {
	document := strictV03WithGlobal("  modules:\n    prompt_guard:\n      model_ref: prompt_guard\n")
	_, err := testAuthoringParser(t).ParseYAMLBytes(document)
	if err == nil || !strings.Contains(err.Error(), "global") || !strings.Contains(err.Error(), "modules") {
		t.Fatalf("expected global.modules rejection, got: %v", err)
	}
}

func TestParseYAMLBytesRejectsUnknownEmbeddingCatalogField(t *testing.T) {
	document := strictV03WithGlobal("  model_catalog:\n    embeddings:\n      bert:\n        model_id: old-bert\n")
	_, err := testAuthoringParser(t).ParseYAMLBytes(document)
	if err == nil || !strings.Contains(err.Error(), "field bert") {
		t.Fatalf("expected deprecated embeddings.bert rejection, got: %v", err)
	}
}

func TestParseYAMLBytesRejectsDeprecatedDecisionModelSelectionAlgorithmField(t *testing.T) {
	document := []byte(strings.Replace(
		strictV03AuthoringYAML,
		"          rules: {}",
		"          rules: {}\n          modelSelectionAlgorithm: {enabled: true, method: router_dc}",
		1,
	))
	_, err := testAuthoringParser(t).ParseYAMLBytes(document)
	if err == nil || !strings.Contains(err.Error(), "modelSelectionAlgorithm") {
		t.Fatalf("expected deprecated decision field rejection, got: %v", err)
	}
}

func TestParseYAMLBytesParsesNestedCanonicalGlobalModules(t *testing.T) {
	document := strictV03WithGlobal(`  router:
    clear_route_cache: false
    streamed_body: {enabled: true, max_bytes: 4096, timeout_sec: 12}
  stores:
    response_cache: {enabled: false}
  model_catalog:
    embeddings:
      semantic:
        qwen3_model_path: models/mom-embedding-pro
        bert_model_path: models/mom-embedding-light
        use_cpu: true
        embedding_config: {min_score_threshold: 0.6}
    system:
      prompt_guard: models/custom-jailbreak
    modules:
      prompt_guard: {enabled: true, model_ref: prompt_guard, threshold: 0.8}
`)
	cfg, err := testAuthoringParser(t).ParseYAMLBytes(document)
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}
	if cfg.ClearRouteCache {
		t.Fatalf("router overrides were not applied: %+v", cfg.RouterOptions)
	}
	if !cfg.StreamedBodyMode || cfg.MaxStreamedBodyBytes != 4096 || cfg.StreamedBodyTimeoutSec != 12 {
		t.Fatalf("streamed body override was not applied: %+v", cfg.RouterOptions)
	}
	if cfg.Enabled {
		t.Fatal("expected response cache override to disable the cache")
	}
	if cfg.PromptGuard.ModelID != "models/custom-jailbreak" || cfg.Qwen3ModelPath != "models/mom-embedding-pro" || cfg.BertModelPath != "models/mom-embedding-light" {
		t.Fatalf("model catalog overrides were not applied: prompt_guard=%+v qwen=%q bert=%q", cfg.PromptGuard, cfg.Qwen3ModelPath, cfg.BertModelPath)
	}
	if cfg.ModelConfig["model-c"].ReasoningFamily == "" {
		t.Fatal("native Model reasoning family was not compiled")
	}
}

func TestParseYAMLBytesPreservesGlobalServiceDefaultsForSparseOverrides(t *testing.T) {
	document := strictV03WithGlobal(`  stores:
    memory: {enabled: true, auto_store: true}
  model_catalog:
    embeddings:
      semantic: {bert_model_path: models/mom-embedding-light, use_cpu: true}
`)
	cfg, err := testAuthoringParser(t).ParseYAMLBytes(document)
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}
	if !cfg.ResponseAPI.Enabled || cfg.ResponseAPI.StoreBackend != "memory" || cfg.ResponseAPI.TTLSeconds != 86400 {
		t.Fatalf("sparse override lost response API defaults: %+v", cfg.ResponseAPI)
	}
	if cfg.RouterReplay.Enabled || cfg.RouterReplay.StoreBackend != "memory" || cfg.RouterReplay.TTLSeconds != 2592000 {
		t.Fatalf("sparse override changed router replay defaults: %+v", cfg.RouterReplay)
	}
	if !cfg.Memory.Enabled || !cfg.Memory.AutoStore {
		t.Fatalf("memory override was not applied: %+v", cfg.Memory)
	}
}

func TestParseYAMLBytesPreservesDefaultSystemModelsForSparseModuleOverrides(t *testing.T) {
	document := strictV03WithGlobal(`  model_catalog:
    modules:
      classifier:
        domain: {threshold: 0.6, use_cpu: true, model_ref: domain_classifier}
        pii: {threshold: 0.7, use_cpu: true, model_ref: pii_classifier}
      prompt_guard: {enabled: true, threshold: 0.7, use_cpu: true, model_ref: prompt_guard}
`)
	cfg, err := testAuthoringParser(t).ParseYAMLBytes(document)
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}
	if cfg.CategoryModel.ModelID != "models/mmbert32k-intent-classifier-merged" || !cfg.CategoryModel.UseMmBERT32K {
		t.Fatalf("domain classifier defaults were not preserved: %+v", cfg.CategoryModel)
	}
	if cfg.PIIModel.ModelID != "models/mmbert32k-pii-detector-merged" || !cfg.PIIModel.UseMmBERT32K {
		t.Fatalf("PII defaults were not preserved: %+v", cfg.PIIModel)
	}
	if cfg.PromptGuard.ModelID != "models/mmbert32k-jailbreak-detector-merged" || cfg.PromptGuard.Variant != PromptGuardVariantMmBERT32K {
		t.Fatalf("prompt guard defaults were not preserved: %+v", cfg.PromptGuard)
	}
	if !cfg.PreferenceModel.ContrastiveEnabled() {
		t.Fatal("preference classifier default mode was not preserved")
	}
}

func TestParseYAMLBytesPreservesNativeModelPricing(t *testing.T) {
	document := strings.Replace(strictV03AuthoringYAML, "global:\n", "global:\n  billing:\n    currency: USD\n", 1)
	document = strings.Replace(document, "    - name: model-a\n", `    - name: model-a
      pricing:
        input_cost_per_million_tokens: "0.24"
        output_cost_per_million_tokens: "0.96"
        cache_read_cost_per_million_tokens: "0.06"
        cache_write_cost_per_million_tokens: "0.30"
`, 1)
	cfg, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}
	pricing := cfg.ModelConfig["model-a"].RuntimePricing
	if pricing.InputCostPerMillionTokens == nil || *pricing.InputCostPerMillionTokens != "0.24" ||
		pricing.OutputCostPerMillionTokens == nil || *pricing.OutputCostPerMillionTokens != "0.96" ||
		pricing.CacheReadCostPerMillionTokens == nil || *pricing.CacheReadCostPerMillionTokens != "0.06" ||
		pricing.CacheWriteCostPerMillionTokens == nil || *pricing.CacheWriteCostPerMillionTokens != "0.3" ||
		cfg.BillingCurrency != "USD" {
		t.Fatalf("native Model pricing was not preserved: currency=%q pricing=%+v", cfg.BillingCurrency, pricing)
	}
}

func TestGetModelPricingTreatsExplicitZeroPricingAsConfigured(t *testing.T) {
	cfg := &RouterConfig{BackendModels: BackendModels{ModelConfig: map[string]ModelParams{
		"qwen-rocm": {Pricing: ModelPricing{Currency: "USD"}},
	}}}
	prompt, completion, currency, ok := cfg.GetModelPricing("qwen-rocm")
	if !ok || prompt != 0 || completion != 0 || currency != "USD" {
		t.Fatalf("zero pricing lookup = (%v, %v, %q, %v)", prompt, completion, currency, ok)
	}
}

func TestGetModelPricingResolvesExternalModelID(t *testing.T) {
	cfg := &RouterConfig{BackendModels: BackendModels{ModelConfig: map[string]ModelParams{
		"claude": {
			Pricing:          ModelPricing{Currency: "USD", PromptPer1M: 5.5, CompletionPer1M: 27.5},
			ExternalModelIDs: map[string]string{"default": "provider/claude"},
		},
	}}}
	for _, name := range []string{"claude", "provider/claude"} {
		prompt, completion, currency, ok := cfg.GetModelPricing(name)
		if !ok || prompt != 5.5 || completion != 27.5 || currency != "USD" {
			t.Fatalf("pricing lookup for %q = (%v, %v, %q, %v)", name, prompt, completion, currency, ok)
		}
	}
	if _, _, _, ok := cfg.GetModelPricing("missing"); ok {
		t.Fatal("unknown model unexpectedly had pricing")
	}
}

func TestParseYAMLBytesAllowsClearingRouterOwnedClassifierDefaults(t *testing.T) {
	document := strictV03WithGlobal(`  model_catalog:
    modules:
      prompt_guard:
        enabled: false
        model_ref: ""
        model_id: ""
        jailbreak_mapping_path: ""
      classifier:
        domain: {model_ref: "", model_id: "", category_mapping_path: "", use_mmbert_32k: false}
        pii: {model_ref: "", model_id: "", pii_mapping_path: "", use_mmbert_32k: false}
`)
	cfg, err := testAuthoringParser(t).ParseYAMLBytes(document)
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}
	if cfg.CategoryModel.ModelID != "" || cfg.CategoryMappingPath != "" || cfg.CategoryModel.UseMmBERT32K {
		t.Fatalf("domain classifier defaults were not cleared: %+v", cfg.CategoryModel)
	}
	if cfg.PIIModel.ModelID != "" || cfg.PIIMappingPath != "" || cfg.PIIModel.UseMmBERT32K {
		t.Fatalf("PII classifier defaults were not cleared: %+v", cfg.PIIModel)
	}
	if cfg.PromptGuard.Enabled || cfg.PromptGuard.ModelID != "" || cfg.PromptGuard.JailbreakMappingPath != "" {
		t.Fatalf("prompt guard defaults were not cleared: %+v", cfg.PromptGuard)
	}
}

func TestParseYAMLBytesParsesCanonicalLoRACatalog(t *testing.T) {
	document := strings.Replace(
		strictV03AuthoringYAML,
		"    - {name: model-a}\n",
		"    - name: model-a\n      loras: [{name: sql-expert}, {name: code-review}]\n",
		1,
	)
	document = strings.Replace(document, "      finish: {models: [{model: model-a}]}", "      finish: {models: [{model: model-a, lora: sql-expert}]}", 1)
	cfg, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}
	loras := cfg.ModelConfig["model-a"].LoRAs
	if len(loras) != 2 || loras[0].Name != "code-review" || loras[1].Name != "sql-expert" {
		t.Fatalf("unexpected LoRA catalog: %#v", loras)
	}
	recipe, found := cfg.RecipeForRequestModel("vllm-sr/edge")
	if !found || recipe.Profile.Decisions[1].ModelRefs[0].LoRAName != "sql-expert" {
		t.Fatalf("LoRA assignment was not compiled: %+v", recipe)
	}
}
