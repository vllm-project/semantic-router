package config

import "testing"

func TestParseYAMLBytesAllowsLoRADefaultModelAlias(t *testing.T) {
	canonicalYAML := []byte(`
version: v0.3
listeners: []
providers:
  defaults:
    default_model: general-expert
    reasoning_families:
      qwen3:
        type: chat_template_kwargs
        parameter: enable_thinking
  models:
    - name: base-model
      reasoning_family: qwen3
      backend_refs:
        - endpoint: 127.0.0.1:8000
          api_key: alias-secret
routing:
  modelCards:
    - name: base-model
      loras:
        - name: general-expert
          description: Default LoRA adapter
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
        - model: base-model
          lora_name: general-expert
`)

	cfg, err := ParseYAMLBytes(canonicalYAML)
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}
	if cfg.DefaultModel != "general-expert" {
		t.Fatalf("expected default model to stay on LoRA alias, got %q", cfg.DefaultModel)
	}

	endpoints := cfg.GetEndpointsForModel("general-expert")
	if len(endpoints) != 1 {
		t.Fatalf("expected LoRA alias to inherit one endpoint, got %#v", endpoints)
	}
	if endpoints[0].Name != "base-model_primary" {
		t.Fatalf("expected LoRA alias endpoint name to inherit base endpoint, got %#v", endpoints[0])
	}

	address, endpointName, found, detailErr := cfg.SelectBestEndpointWithDetailsForModel("general-expert")
	if detailErr != nil {
		t.Fatalf("SelectBestEndpointWithDetailsForModel returned error: %v", detailErr)
	}
	if !found {
		t.Fatal("expected LoRA alias endpoint to resolve")
	}
	if address != "127.0.0.1:8000" || endpointName != "base-model_primary" {
		t.Fatalf("unexpected endpoint resolution for LoRA alias: address=%q endpoint=%q", address, endpointName)
	}

	family := cfg.GetModelReasoningFamily("general-expert")
	if family == nil || family.Parameter != "enable_thinking" {
		t.Fatalf("expected LoRA alias to inherit reasoning family, got %#v", family)
	}
	if accessKey := cfg.GetModelAccessKey("general-expert"); accessKey != "alias-secret" {
		t.Fatalf("expected LoRA alias to inherit access key, got %q", accessKey)
	}
}

func TestGetEndpointsForModelFallsBackWhenAliasModelCardHasNoProviderMetadata(t *testing.T) {
	canonicalYAML := []byte(`
version: v0.3
listeners: []
providers:
  defaults:
    default_model: general-expert
  models:
    - name: base-model
      backend_refs:
        - endpoint: 127.0.0.1:8000
routing:
  modelCards:
    - name: base-model
      loras:
        - name: general-expert
          description: Default LoRA adapter
    - name: general-expert
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
        - model: base-model
          lora_name: general-expert
`)

	cfg, err := ParseYAMLBytes(canonicalYAML)
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}

	endpoints := cfg.GetEndpointsForModel("general-expert")
	if len(endpoints) != 1 {
		t.Fatalf("expected alias model card without provider metadata to inherit base endpoints, got %#v", endpoints)
	}
	if endpoints[0].Name != "base-model_primary" {
		t.Fatalf("unexpected fallback endpoint for alias model card: %#v", endpoints[0])
	}
}
