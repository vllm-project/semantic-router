package config

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

func categoryBackendTestConfig(backend *RemoteClassifierBackend) *RouterConfig {
	return &RouterConfig{
		InlineModels: InlineModels{Classifier: Classifier{CategoryModel: CategoryModel{
			ModelID:             "local-category",
			CategoryMappingPath: "models/category.json",
			Backend:             backend,
		}}},
		ExternalModels: []ExternalModelConfig{{
			Name:      "named-category",
			ModelRole: ModelRoleClassification,
			ModelName: "category-service",
			ModelEndpoint: ClassifierVLLMEndpoint{
				Address: "127.0.0.1",
				Port:    8080,
			},
		}},
	}
}

func TestValidateCategoryModelBackend(t *testing.T) {
	deadline := 3000
	valid := &RemoteClassifierBackend{
		Protocol:   RemoteClassifierProtocolHTTPClassify,
		Model:      "named-category",
		DeadlineMs: &deadline,
	}
	if err := ValidateCategoryModelBackend(categoryBackendTestConfig(valid)); err != nil {
		t.Fatalf("valid named backend rejected: %v", err)
	}

	tests := []struct {
		name   string
		mutate func(*RouterConfig)
		want   string
	}{
		{name: "missing named model", mutate: func(cfg *RouterConfig) {
			cfg.CategoryModel.Backend.Model = "missing"
		}, want: "not declared"},
		{name: "duplicate named model", mutate: func(cfg *RouterConfig) {
			cfg.ExternalModels = append(cfg.ExternalModels, cfg.ExternalModels[0])
		}, want: "ambiguous"},
		{name: "wrong role", mutate: func(cfg *RouterConfig) {
			cfg.ExternalModels[0].ModelRole = ModelRoleGuardrail
		}, want: "model_role"},
		{name: "unsupported protocol", mutate: func(cfg *RouterConfig) {
			cfg.CategoryModel.Backend.Protocol = RemoteClassifierProtocolHTTPChat
		}, want: "not supported"},
		{name: "incorrect contract", mutate: func(cfg *RouterConfig) {
			cfg.CategoryModel.Backend.Contract = "label_distribution.v0"
		}, want: "unsupported"},
		{name: "invalid timeout", mutate: func(cfg *RouterConfig) {
			zero := 0
			cfg.CategoryModel.Backend.DeadlineMs = &zero
		}, want: "deadline_ms"},
		{name: "invalid endpoint port", mutate: func(cfg *RouterConfig) {
			cfg.ExternalModels[0].ModelEndpoint.Port = 65536
		}, want: "valid llm_endpoint"},
		{name: "invalid endpoint protocol", mutate: func(cfg *RouterConfig) {
			cfg.ExternalModels[0].ModelEndpoint.Protocol = "ftp"
		}, want: "http or https"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := categoryBackendTestConfig(&RemoteClassifierBackend{
				Protocol: RemoteClassifierProtocolHTTPClassify,
				Model:    "named-category",
			})
			tt.mutate(cfg)
			err := ValidateCategoryModelBackend(cfg)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want substring %q", err, tt.want)
			}
		})
	}
}

func TestRemoteClassifierBackendYAMLDefaultsRemainOmitted(t *testing.T) {
	var cfg struct {
		Backend *RemoteClassifierBackend `yaml:"backend"`
	}
	if err := yaml.Unmarshal([]byte(`backend:
  protocol: http_classify
  model: named-category
`), &cfg); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if cfg.Backend == nil || cfg.Backend.Contract != "" || cfg.Backend.DeadlineMs != nil {
		t.Fatalf("omitted defaults lost: %#v", cfg.Backend)
	}
	if got := cfg.Backend.EffectiveContract(RemoteClassifierContractLabelDistribution); got != RemoteClassifierContractLabelDistribution {
		t.Fatalf("omitted contract = %q, want %q", got, RemoteClassifierContractLabelDistribution)
	}
	if got := cfg.Backend.EffectiveDeadlineMs(); got != defaultRemoteClassifierDeadlineMs {
		t.Fatalf("omitted deadline = %d, want %d", got, defaultRemoteClassifierDeadlineMs)
	}
	encoded, err := yaml.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(encoded), "deadline_ms") || !strings.Contains(string(encoded), "protocol: http_classify") {
		t.Fatalf("unexpected backend serialization: %s", encoded)
	}
}

func TestRemoteClassifierBackendYAMLExplicitFieldsRoundTrip(t *testing.T) {
	var canonical struct {
		Backend *RemoteClassifierBackend `yaml:"backend"`
	}
	if err := yaml.Unmarshal([]byte(`backend:
  protocol: http_classify
  model: named-category
  contract: label_distribution.v1
  deadline_ms: 7000
`), &canonical); err != nil {
		t.Fatalf("canonical backend unmarshal: %v", err)
	}
	if canonical.Backend == nil || canonical.Backend.DeadlineMs == nil || *canonical.Backend.DeadlineMs != 7000 {
		t.Fatalf("canonical deadline was not preserved: %#v", canonical.Backend)
	}
	encodedCanonical, err := yaml.Marshal(canonical)
	if err != nil || !strings.Contains(string(encodedCanonical), "deadline_ms: 7000") || strings.Contains(string(encodedCanonical), "timeout_seconds") {
		t.Fatalf("canonical backend serialization = %s, err=%v", encodedCanonical, err)
	}
}

func TestRemoteClassifierBackendYAMLRejectsTimeoutAlias(t *testing.T) {
	var stale struct {
		Backend *RemoteClassifierBackend `yaml:"backend"`
	}
	if err := yaml.Unmarshal([]byte(`backend:
  protocol: http_classify
  model: named-category
  timeout_seconds: 5
`), &stale); err == nil || !strings.Contains(err.Error(), "deadline_ms") {
		t.Fatalf("expected stale timeout_seconds to be rejected, got %v", err)
	}
}

func TestReferenceConfigCategoryBackendReplacesDefaultVariant(t *testing.T) {
	data := string(readReferenceConfigYAML(t))
	data = strings.Replace(data,
		"          category_mapping_path: \"\"\n",
		"          backend:\n"+
			"            protocol: http_classify\n"+
			"            contract: label_distribution.v1\n"+
			"            model: external-classifier\n"+
			"            deadline_ms: 5000\n"+
			"          category_mapping_path: models/Vela-1.0-Encoder-307M-Domain/category_mapping.json\n", 1)
	if data == string(readReferenceConfigYAML(t)) {
		t.Fatal("reference config category block was not found")
	}
	if _, err := ParseYAMLBytes([]byte(data)); err != nil {
		t.Fatalf("reference config plus backend rejected: %v", err)
	}
}

func piiBackendTestConfig(backend *RemoteClassifierBackend) *RouterConfig {
	cfg := &RouterConfig{}
	cfg.PIIModel = PIIModel{Backend: backend, PIIMappingPath: "models/pii/pii_type_mapping.json"}
	cfg.ExternalModels = []ExternalModelConfig{{
		Name:          "named-pii",
		ModelRole:     ModelRoleClassification,
		ModelEndpoint: ClassifierVLLMEndpoint{Address: "10.0.0.5", Port: 8000},
		ModelName:     "pii-spans",
	}}
	return cfg
}

func TestRemoteClassifierBackendAcceptsTokenSpansContract(t *testing.T) {
	b := &RemoteClassifierBackend{Protocol: RemoteClassifierProtocolHTTPClassify, Contract: RemoteClassifierContractTokenSpans, Model: "named-pii"}
	if err := b.Validate(); err != nil {
		t.Fatalf("token_spans.v1 rejected: %v", err)
	}
	if got := b.EffectiveContract(RemoteClassifierContractLabelDistribution); got != RemoteClassifierContractTokenSpans {
		t.Fatalf("explicit contract %q overridden to %q", RemoteClassifierContractTokenSpans, got)
	}
}

func TestValidatePIIModelBackend(t *testing.T) {
	if err := ValidatePIIModelBackend(&RouterConfig{}); err != nil {
		t.Fatalf("absent backend must be a no-op: %v", err)
	}
	valid := &RemoteClassifierBackend{Protocol: RemoteClassifierProtocolHTTPClassify, Model: "named-pii"}
	if err := ValidatePIIModelBackend(piiBackendTestConfig(valid)); err != nil {
		t.Fatalf("valid backend with defaulted contract rejected: %v", err)
	}
	tests := []struct {
		name   string
		mutate func(*RouterConfig)
		want   string
	}{
		{name: "label distribution is the wrong shape for PII", mutate: func(cfg *RouterConfig) {
			cfg.PIIModel.Backend.Contract = RemoteClassifierContractLabelDistribution
		}, want: "incompatible"},
		{name: "unknown contract", mutate: func(cfg *RouterConfig) {
			cfg.PIIModel.Backend.Contract = "token_spans.v0"
		}, want: "unsupported"},
		{name: "chat protocol not supported", mutate: func(cfg *RouterConfig) {
			cfg.PIIModel.Backend.Protocol = RemoteClassifierProtocolHTTPChat
		}, want: "not supported"},
		{name: "wrong role", mutate: func(cfg *RouterConfig) {
			cfg.ExternalModels[0].ModelRole = ModelRoleGuardrail
		}, want: "model_role"},
		{name: "missing named model", mutate: func(cfg *RouterConfig) {
			cfg.PIIModel.Backend.Model = "missing"
		}, want: "not declared"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := piiBackendTestConfig(&RemoteClassifierBackend{Protocol: RemoteClassifierProtocolHTTPClassify, Model: "named-pii"})
			tt.mutate(cfg)
			err := ValidatePIIModelBackend(cfg)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("want error containing %q, got %v", tt.want, err)
			}
		})
	}
}

// classifier.pii.on_error takes the shared allow|block contract and nothing else.
func TestPIIOnErrorValidatedAtParse(t *testing.T) {
	badYAML := []byte(`
version: v0.3
global:
  model_catalog:
    modules:
      classifier:
        pii:
          on_error: retry
`)
	_, err := ParseYAMLBytes(badYAML)
	if err == nil || !strings.Contains(err.Error(), "on_error") {
		t.Fatalf("unknown pii on_error must be rejected at parse, got %v", err)
	}

	goodYAML := []byte(`
version: v0.3
global:
  model_catalog:
    modules:
      classifier:
        pii:
          on_error: block
`)
	cfg, err := ParseYAMLBytes(goodYAML)
	if err != nil {
		t.Fatalf("pii on_error: block rejected: %v", err)
	}
	if !cfg.PIIModel.IsBlock() {
		t.Fatal("pii on_error: block was not decoded onto PIIModel")
	}
}

// A remote-only PII configuration must still be reported as enabled, otherwise
// NeedsPIIMappingForRouting is false and the mapping the adapter requires is
// never loaded.
func TestIsPIIClassifierEnabledAcceptsRemoteBackend(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.PIIModel.Backend = &RemoteClassifierBackend{Protocol: RemoteClassifierProtocolHTTPClassify, Model: "named-pii"}
	cfg.PIIMappingPath = "models/pii/pii_type_mapping.json"
	if !cfg.IsPIIClassifierEnabled() {
		t.Fatal("backend-only PII config reported disabled")
	}
	cfg.PIIModel.Backend = nil
	if cfg.IsPIIClassifierEnabled() {
		t.Fatal("PII config with neither model_id nor backend reported enabled")
	}
}

func TestFixedHTTPClassifierDoesNotRequireAnUnusedModelSelector(t *testing.T) {
	cfg := categoryBackendTestConfig(&RemoteClassifierBackend{Protocol: RemoteClassifierProtocolHTTPClassify, Model: "named-category"})
	cfg.ExternalModels[0].ModelName = ""
	if err := ValidateCategoryModelBackend(cfg); err != nil {
		t.Fatal(err)
	}
	cfg.ExternalModels[0].ModelRole = ModelRoleGuardrail
	backend := &RemoteClassifierBackend{Protocol: RemoteClassifierProtocolHTTPChat, Model: "named-category", Contract: RemoteClassifierContractLabelDecision}
	if _, err := ResolveRemoteClassifierBackend(cfg, backend, ModelRoleGuardrail, RemoteClassifierContractLabelDecision); err == nil {
		t.Fatal("chat request accepted without its model selector")
	}
}
