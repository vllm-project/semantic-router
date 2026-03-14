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
      provider_model_id: served-qwen
      backend_refs:
        - name: primary
          endpoint: 127.0.0.1:11434
          protocol: http
routing:
  modelCards:
    - name: qwen2.5:3b
      reasoning_family_ref: qwen3
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
        use_cpu: true
      bert:
        model_id: models/mom-bert
        threshold: 0.6
        use_cpu: true
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
	if cfg.SemanticCache.Enabled {
		t.Fatal("expected semantic cache enabled override to be false")
	}
	if cfg.PromptGuard.ModelID != "models/custom-jailbreak" {
		t.Fatalf("expected prompt guard model to follow system override, got %q", cfg.PromptGuard.ModelID)
	}
	if cfg.EmbeddingModels.Qwen3ModelPath != "models/mom-embedding-pro" {
		t.Fatalf("expected semantic embedding model override, got %q", cfg.EmbeddingModels.Qwen3ModelPath)
	}
	if cfg.BertModel.ModelID != "models/mom-bert" {
		t.Fatalf("expected bert model override, got %q", cfg.BertModel.ModelID)
	}
	if got := cfg.ModelConfig["qwen2.5:3b"].ReasoningFamily; got != "qwen3" {
		t.Fatalf("expected routing model reasoning family, got %q", got)
	}
	if len(cfg.VLLMEndpoints) != 1 || cfg.VLLMEndpoints[0].Name != "qwen2.5:3b_primary" {
		t.Fatalf("expected canonical provider endpoint to normalize, got %#v", cfg.VLLMEndpoints)
	}
}
