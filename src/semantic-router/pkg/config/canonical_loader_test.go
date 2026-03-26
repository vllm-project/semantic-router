package config

import (
	"strings"
	"testing"
)

func TestParseYAMLBytesRejectsLegacyUserConfigLayout(t *testing.T) {
	legacyYAML := []byte(`
version: v0.3
signals:
  keywords:
    - name: urgent_keywords
      operator: OR
      keywords: ["urgent"]
decisions:
  - name: urgent_route
    rules:
      operator: AND
      conditions:
        - type: keyword
          name: urgent_keywords
    modelRefs:
      - model: qwen2.5:3b
        use_reasoning: false
providers:
  default_model: qwen2.5:3b
  models:
    - name: qwen2.5:3b
      endpoints:
        - endpoint: 127.0.0.1:11434
`)

	_, err := ParseYAMLBytes(legacyYAML)
	if err == nil {
		t.Fatal("expected legacy user config layout to be rejected")
	}

	message := err.Error()
	for _, fragment := range []string{
		"deprecated config fields are no longer supported",
		"providers.default_model",
		"providers.models[0].endpoints",
		"vllm-sr config migrate --config old-config.yaml",
	} {
		if !strings.Contains(message, fragment) {
			t.Fatalf("expected error to mention %q, got: %s", fragment, message)
		}
	}
}

func TestParseYAMLBytesRejectsTopLevelLegacyRuntimeLayout(t *testing.T) {
	legacyYAML := []byte(`
version: v0.3
default_model: qwen2.5:3b
semantic_cache:
  enabled: false
`)

	_, err := ParseYAMLBytes(legacyYAML)
	if err == nil {
		t.Fatal("expected top-level legacy runtime layout to be rejected")
	}

	message := err.Error()
	for _, fragment := range []string{
		"config file must use canonical v0.3 version/listeners/providers/routing/global",
		"unexpected top-level keys: default_model, semantic_cache",
		"vllm-sr config migrate --config old-config.yaml",
	} {
		if !strings.Contains(message, fragment) {
			t.Fatalf("expected error to mention %q, got: %s", fragment, message)
		}
	}
}

func TestParseYAMLBytesRejectsDeprecatedGlobalModulesLayout(t *testing.T) {
	canonicalYAML := []byte(`
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    default_model: qwen2.5:3b
  models:
    - name: qwen2.5:3b
      backend_refs:
        - endpoint: 127.0.0.1:11434
routing:
  modelCards:
    - name: qwen2.5:3b
  decisions:
    - name: default-route
      description: fallback
      priority: 100
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: qwen2.5:3b
global:
  modules:
    prompt_guard:
      model_ref: prompt_guard
`)

	_, err := ParseYAMLBytes(canonicalYAML)
	if err == nil {
		t.Fatal("expected deprecated global.modules layout to be rejected")
	}
	if !strings.Contains(err.Error(), "global.modules") {
		t.Fatalf("expected error to mention global.modules, got: %v", err)
	}
}

func TestParseYAMLBytesRejectsDeprecatedGlobalModelCatalogEmbeddingsBertLayout(t *testing.T) {
	canonicalYAML := []byte(`
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    default_model: qwen2.5:3b
  models:
    - name: qwen2.5:3b
      backend_refs:
        - endpoint: 127.0.0.1:11434
routing:
  modelCards:
    - name: qwen2.5:3b
  decisions:
    - name: default-route
      description: fallback
      priority: 100
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: qwen2.5:3b
global:
  model_catalog:
    embeddings:
      bert:
        model_id: models/mom-embedding-light
        threshold: 0.6
        use_cpu: true
`)

	_, err := ParseYAMLBytes(canonicalYAML)
	if err == nil {
		t.Fatal("expected deprecated global.model_catalog.embeddings.bert layout to be rejected")
	}
	if !strings.Contains(err.Error(), "global.model_catalog.embeddings.bert") {
		t.Fatalf("expected error to mention global.model_catalog.embeddings.bert, got: %v", err)
	}
}

func TestParseYAMLBytesRejectsDeprecatedDecisionModelSelectionAlgorithmField(t *testing.T) {
	canonicalYAML := []byte(`
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    default_model: qwen2.5:3b
  models:
    - name: qwen2.5:3b
      backend_refs:
        - endpoint: 127.0.0.1:11434
routing:
  modelCards:
    - name: qwen2.5:3b
  decisions:
    - name: support-route
      description: fallback
      priority: 100
      rules:
        operator: AND
        conditions: []
      modelSelectionAlgorithm:
        enabled: true
        method: router_dc
      modelRefs:
        - model: qwen2.5:3b
          use_reasoning: false
global:
  router:
    strategy: priority
`)

	_, err := ParseYAMLBytes(canonicalYAML)
	if err == nil {
		t.Fatal("expected deprecated decision modelSelectionAlgorithm field to be rejected")
	}
	if !strings.Contains(err.Error(), "routing.decisions[0].modelSelectionAlgorithm") {
		t.Fatalf("expected error to mention deprecated decision field, got: %v", err)
	}
}

func TestParseYAMLBytesParsesNestedCanonicalModelCatalogModules(t *testing.T) {
	canonicalYAML := []byte(`
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    default_model: qwen2.5:3b
    default_reasoning_effort: low
    reasoning_families:
      qwen3:
        type: chat_template_kwargs
        parameter: enable_thinking
  models:
    - name: qwen2.5:3b
      reasoning_family: qwen3
      provider_model_id: served-qwen
      backend_refs:
        - name: primary
          endpoint: 127.0.0.1:11434
          protocol: http
routing:
  modelCards:
    - name: qwen2.5:3b
      param_size: 3b
global:
  router:
    auto_model_name: auto
    clear_route_cache: false
    streamed_body:
      enabled: true
      max_bytes: 4096
      timeout_sec: 12
  stores:
    semantic_cache:
      enabled: false
  model_catalog:
    embeddings:
      semantic:
        qwen3_model_path: models/mom-embedding-pro
        bert_model_path: models/mom-embedding-light
        use_cpu: true
        embedding_config:
          min_score_threshold: 0.6
    system:
      prompt_guard: models/custom-jailbreak
    modules:
      prompt_guard:
        enabled: true
        model_ref: prompt_guard
        threshold: 0.8
`)

	cfg, err := ParseYAMLBytes(canonicalYAML)
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}

	if cfg.DefaultModel != "qwen2.5:3b" {
		t.Fatalf("expected default model to be preserved, got %q", cfg.DefaultModel)
	}
	if cfg.DefaultReasoningEffort != "low" {
		t.Fatalf("expected default reasoning effort to be preserved, got %q", cfg.DefaultReasoningEffort)
	}
	if cfg.AutoModelName != "auto" {
		t.Fatalf("expected auto model name override, got %q", cfg.AutoModelName)
	}
	if cfg.ClearRouteCache {
		t.Fatal("expected clear_route_cache override to be false")
	}
	if !cfg.StreamedBodyMode || cfg.MaxStreamedBodyBytes != 4096 || cfg.StreamedBodyTimeoutSec != 12 {
		t.Fatalf("expected streamed body runtime override, got enabled=%v max=%d timeout=%d", cfg.StreamedBodyMode, cfg.MaxStreamedBodyBytes, cfg.StreamedBodyTimeoutSec)
	}
	if cfg.Enabled {
		t.Fatal("expected semantic cache enabled override to be false")
	}
	if cfg.PromptGuard.ModelID != "models/custom-jailbreak" {
		t.Fatalf("expected prompt guard model to follow system override, got %q", cfg.PromptGuard.ModelID)
	}
	if cfg.Qwen3ModelPath != "models/mom-embedding-pro" {
		t.Fatalf("expected semantic embedding model override, got %q", cfg.Qwen3ModelPath)
	}
	if cfg.BertModelPath != "models/mom-embedding-light" {
		t.Fatalf("expected bert embedding path override, got %q", cfg.BertModelPath)
	}
	if got := cfg.ModelConfig["qwen2.5:3b"].ReasoningFamily; got != "qwen3" {
		t.Fatalf("expected provider model reasoning family, got %q", got)
	}
	if len(cfg.VLLMEndpoints) != 1 || cfg.VLLMEndpoints[0].Name != "qwen2.5:3b_primary" {
		t.Fatalf("expected canonical provider endpoint to normalize, got %#v", cfg.VLLMEndpoints)
	}
}

func TestParseYAMLBytesPreservesGlobalServiceDefaultsForSparseCanonicalOverrides(t *testing.T) {
	canonicalYAML := []byte(`
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8888
providers:
  defaults:
    default_model: qwen3
  models:
    - name: qwen3
      provider_model_id: qwen3
      backend_refs:
        - endpoint: 127.0.0.1:8000
routing:
  modelCards:
    - name: qwen3
      modality: text
  decisions:
    - name: default_route
      priority: 1
      rules:
        operator: OR
        conditions:
          - type: domain
            name: general
      modelRefs:
        - model: qwen3
global:
  stores:
    memory:
      enabled: true
      auto_store: true
  model_catalog:
    embeddings:
      semantic:
        bert_model_path: models/mom-embedding-light
        use_cpu: true
`)

	cfg, err := ParseYAMLBytes(canonicalYAML)
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}

	if !cfg.ResponseAPI.Enabled {
		t.Fatal("expected sparse global override to preserve default response_api.enabled=true")
	}
	if cfg.ResponseAPI.StoreBackend != "memory" {
		t.Fatalf("expected response api backend to keep default, got %q", cfg.ResponseAPI.StoreBackend)
	}
	if cfg.ResponseAPI.TTLSeconds != 86400 {
		t.Fatalf("expected response api ttl default to be preserved, got %d", cfg.ResponseAPI.TTLSeconds)
	}
	if !cfg.Memory.Enabled || !cfg.Memory.AutoStore {
		t.Fatalf("expected memory override to still apply, got enabled=%v auto_store=%v", cfg.Memory.Enabled, cfg.Memory.AutoStore)
	}
}

func TestParseYAMLBytesPreservesDefaultSystemModelsForSparseModuleOverrides(t *testing.T) {
	canonicalYAML := []byte(`
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8888
providers:
  defaults:
    default_model: qwen3
  models:
    - name: qwen3
      provider_model_id: qwen3
      backend_refs:
        - endpoint: 127.0.0.1:8000
routing:
  signals:
    domains:
      - name: general
        description: General requests
        mmlu_categories: [other]
  modelCards:
    - name: qwen3
  decisions:
    - name: default_route
      priority: 1
      rules:
        operator: OR
        conditions:
          - type: domain
            name: general
      modelRefs:
        - model: qwen3
global:
  model_catalog:
    modules:
      classifier:
        domain:
          threshold: 0.6
          use_cpu: true
          model_ref: domain_classifier
        pii:
          threshold: 0.7
          use_cpu: true
          model_ref: pii_classifier
      prompt_guard:
        enabled: true
        threshold: 0.7
        use_cpu: true
        model_ref: prompt_guard
`)

	cfg, err := ParseYAMLBytes(canonicalYAML)
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}

	if cfg.CategoryModel.ModelID != "models/mmbert32k-intent-classifier-merged" {
		t.Fatalf("expected sparse category override to keep default system model, got %q", cfg.CategoryModel.ModelID)
	}
	if !cfg.CategoryModel.UseMmBERT32K {
		t.Fatal("expected sparse category override to keep mmBERT-32K enabled")
	}
	if cfg.PIIModel.ModelID != "models/mmbert32k-pii-detector-merged" {
		t.Fatalf("expected sparse PII override to keep default system model, got %q", cfg.PIIModel.ModelID)
	}
	if !cfg.PIIModel.UseMmBERT32K {
		t.Fatal("expected sparse PII override to keep mmBERT-32K enabled")
	}
	if cfg.PromptGuard.ModelID != "models/mmbert32k-jailbreak-detector-merged" {
		t.Fatalf("expected sparse prompt-guard override to keep default system model, got %q", cfg.PromptGuard.ModelID)
	}
	if !cfg.PromptGuard.UseMmBERT32K {
		t.Fatal("expected sparse prompt-guard override to keep mmBERT-32K enabled")
	}
	if !cfg.Classifier.PreferenceModel.ContrastiveEnabled() {
		t.Fatal("expected sparse classifier override to preserve default preference contrastive mode")
	}
}

func TestParseYAMLBytesPreservesProviderModelPricing(t *testing.T) {
	canonicalYAML := []byte(`
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8888
providers:
  defaults:
    default_model: qwen3
  models:
    - name: qwen3
      provider_model_id: qwen3
      pricing:
        currency: USD
        prompt_per_1m: 0.24
        completion_per_1m: 0.96
      backend_refs:
        - endpoint: 127.0.0.1:8000
routing:
  modelCards:
    - name: qwen3
      modality: text
  decisions:
    - name: default_route
      priority: 1
      rules:
        operator: OR
        conditions:
          - type: domain
            name: general
      modelRefs:
        - model: qwen3
`)

	cfg, err := ParseYAMLBytes(canonicalYAML)
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}

	pricing := cfg.ModelConfig["qwen3"].Pricing
	if pricing.PromptPer1M != 0.24 || pricing.CompletionPer1M != 0.96 || pricing.Currency != "USD" {
		t.Fatalf("expected provider pricing to be preserved in model config, got %#v", pricing)
	}

	promptPer1M, completionPer1M, currency, ok := cfg.GetModelPricing("qwen3")
	if !ok {
		t.Fatal("expected GetModelPricing to resolve provider pricing")
	}
	if promptPer1M != 0.24 || completionPer1M != 0.96 || currency != "USD" {
		t.Fatalf(
			"expected GetModelPricing to return provider pricing, got prompt=%v completion=%v currency=%q",
			promptPer1M,
			completionPer1M,
			currency,
		)
	}
}

func TestGetModelPricingTreatsExplicitZeroPricingAsConfigured(t *testing.T) {
	cfg := &RouterConfig{
		BackendModels: BackendModels{
			ModelConfig: map[string]ModelParams{
				"qwen-rocm": {
					Pricing: ModelPricing{
						Currency:        "USD",
						PromptPer1M:     0,
						CompletionPer1M: 0,
					},
				},
			},
		},
	}

	promptPer1M, completionPer1M, currency, ok := cfg.GetModelPricing("qwen-rocm")
	if !ok {
		t.Fatal("expected explicit zero pricing to be treated as configured")
	}
	if promptPer1M != 0 || completionPer1M != 0 || currency != "USD" {
		t.Fatalf(
			"expected zero pricing with USD currency, got prompt=%v completion=%v currency=%q",
			promptPer1M,
			completionPer1M,
			currency,
		)
	}
}

func TestParseYAMLBytesAppliesCanonicalRouterConfigSource(t *testing.T) {
	canonicalYAML := []byte(`
version: v0.3
listeners: []
providers:
  defaults: {}
routing:
  signals: {}
global:
  router:
    config_source: kubernetes
`)

	cfg, err := ParseYAMLBytes(canonicalYAML)
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}
	if cfg.ConfigSource != ConfigSourceKubernetes {
		t.Fatalf("expected ConfigSourceKubernetes, got %q", cfg.ConfigSource)
	}
}

func TestParseYAMLBytesAllowsClearingRouterOwnedClassifierDefaults(t *testing.T) {
	canonicalYAML := []byte(`
version: v0.3
listeners: []
providers:
  defaults:
    default_model: openai/gpt-oss-20b
  models:
    - name: openai/gpt-oss-20b
      backend_refs:
        - name: primary
          endpoint: localhost:8000
          protocol: http
          weight: 1
routing:
  decisions:
    - name: default-route
      priority: 100
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: openai/gpt-oss-20b
  modelCards:
    - name: openai/gpt-oss-20b
global:
  model_catalog:
    modules:
      prompt_guard:
        enabled: false
        model_ref: ""
        model_id: ""
        jailbreak_mapping_path: ""
        use_mmbert_32k: false
      classifier:
        domain:
          model_ref: ""
          model_id: ""
          category_mapping_path: ""
          use_mmbert_32k: false
        pii:
          model_ref: ""
          model_id: ""
          pii_mapping_path: ""
          use_mmbert_32k: false
`)

	cfg, err := ParseYAMLBytes(canonicalYAML)
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}

	if cfg.CategoryModel.ModelID != "" {
		t.Fatalf("expected domain classifier model to be cleared, got %q", cfg.CategoryModel.ModelID)
	}
	if cfg.CategoryMappingPath != "" {
		t.Fatalf("expected category mapping path to be cleared, got %q", cfg.CategoryMappingPath)
	}
	if cfg.CategoryModel.UseMmBERT32K {
		t.Fatal("expected domain classifier mmBERT-32K default to be disabled")
	}
	if cfg.PIIModel.ModelID != "" {
		t.Fatalf("expected PII classifier model to be cleared, got %q", cfg.PIIModel.ModelID)
	}
	if cfg.PIIMappingPath != "" {
		t.Fatalf("expected PII mapping path to be cleared, got %q", cfg.PIIMappingPath)
	}
	if cfg.PIIModel.UseMmBERT32K {
		t.Fatal("expected PII classifier mmBERT-32K default to be disabled")
	}
	if cfg.PromptGuard.Enabled {
		t.Fatal("expected prompt guard to be disabled")
	}
	if cfg.PromptGuard.ModelID != "" {
		t.Fatalf("expected prompt guard model to be cleared, got %q", cfg.PromptGuard.ModelID)
	}
	if cfg.PromptGuard.JailbreakMappingPath != "" {
		t.Fatalf("expected prompt guard mapping path to be cleared, got %q", cfg.PromptGuard.JailbreakMappingPath)
	}
}

func TestParseYAMLBytesParsesCanonicalLoRACatalog(t *testing.T) {
	canonicalYAML := []byte(`
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8888
providers:
  defaults:
    default_model: qwen3
  models:
    - name: qwen3
      backend_refs:
        - endpoint: 127.0.0.1:8000
routing:
  modelCards:
    - name: qwen3
      loras:
        - name: sql-expert
          description: SQL-specialized adapter
        - name: code-review
  signals:
    domains:
      - name: other
        description: fallback
  decisions:
    - name: default_route
      priority: 1
      rules:
        operator: AND
        conditions:
          - type: domain
            name: other
      modelRefs:
        - model: qwen3
          lora_name: sql-expert
`)

	cfg, err := ParseYAMLBytes(canonicalYAML)
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}

	loras := cfg.ModelConfig["qwen3"].LoRAs
	if len(loras) != 2 {
		t.Fatalf("expected 2 LoRA adapters, got %#v", loras)
	}
	if loras[0].Name != "sql-expert" || loras[0].Description != "SQL-specialized adapter" {
		t.Fatalf("unexpected first LoRA adapter: %#v", loras[0])
	}
	if cfg.Decisions[0].ModelRefs[0].LoRAName != "sql-expert" {
		t.Fatalf("expected lora_name to survive parse, got %q", cfg.Decisions[0].ModelRefs[0].LoRAName)
	}
}
