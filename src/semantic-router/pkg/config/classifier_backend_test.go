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

func TestCategoryModelLegacySelectorsResolveDeterministically(t *testing.T) {
	tests := []struct {
		name    string
		model   CategoryModel
		variant string
	}{
		{name: "modernbert", model: CategoryModel{UseModernBERT: true}, variant: CategoryVariantModernBERT},
		{name: "mmbert32k", model: CategoryModel{UseMmBERT32K: true}, variant: CategoryVariantMmBERT32K},
		{name: "candle", model: CategoryModel{Variant: CategoryVariantCandle}, variant: CategoryVariantCandle},
		{name: "omitted uses historical auto detect", model: CategoryModel{}, variant: ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.model.EffectiveVariant()
			if err != nil || got != tt.variant {
				t.Fatalf("EffectiveVariant() = %q, %v; want %q", got, err, tt.variant)
			}
		})
	}
}

func TestCategoryModelRejectsContradictoryLocalSelectors(t *testing.T) {
	for name, model := range map[string]CategoryModel{
		"both legacy selectors":         {UseModernBERT: true, UseMmBERT32K: true},
		"modernbert plus mmbert legacy": {UseMmBERT32K: true, Variant: CategoryVariantModernBERT},
		"mmbert plus modern legacy":     {UseModernBERT: true, Variant: CategoryVariantMmBERT32K},
		"candle plus legacy selector":   {UseModernBERT: true, Variant: CategoryVariantCandle},
		"unknown canonical variant":     {Variant: "modern_bert_typo"},
	} {
		t.Run(name, func(t *testing.T) {
			if err := model.ValidateLocalVariant(); err == nil {
				t.Fatal("expected deterministic local-selector validation error")
			}
		})
	}
}

func TestCategoryModelAcceptsAgreeingCanonicalAndLegacySelectors(t *testing.T) {
	for name, model := range map[string]CategoryModel{
		"modernbert":                               {Variant: CategoryVariantModernBERT, UseModernBERT: true},
		"mmbert32k":                                {Variant: CategoryVariantMmBERT32K, UseMmBERT32K: true},
		"modernbert with explicit false mmbert":    {Variant: CategoryVariantModernBERT, UseMmBERT32K: false},
		"mmbert32k with explicit false modernbert": {Variant: CategoryVariantMmBERT32K, UseModernBERT: false},
	} {
		t.Run(name, func(t *testing.T) {
			if err := model.ValidateLocalVariant(); err != nil {
				t.Fatalf("agreeing canonical and legacy selectors rejected: %v", err)
			}
		})
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
		{name: "mixed canonical local selector", mutate: func(cfg *RouterConfig) {
			cfg.CategoryModel.Variant = CategoryVariantMmBERT32K
		}, want: "mutually exclusive"},
		{name: "mixed legacy local selector", mutate: func(cfg *RouterConfig) {
			cfg.CategoryModel.UseModernBERT = true
		}, want: "mutually exclusive"},
		{name: "mixed active legacy local selector", mutate: func(cfg *RouterConfig) {
			cfg.CategoryModel.UseMmBERT32K = true
		}, want: "mutually exclusive"},
		{name: "backend plus agreeing canonical and legacy selector", mutate: func(cfg *RouterConfig) {
			cfg.CategoryModel.Variant = CategoryVariantMmBERT32K
			cfg.CategoryModel.UseMmBERT32K = true
		}, want: "mutually exclusive"},
		{name: "invalid endpoint port", mutate: func(cfg *RouterConfig) {
			cfg.ExternalModels[0].ModelEndpoint.Port = 65536
		}, want: "valid llm_endpoint"},
		{name: "invalid endpoint protocol", mutate: func(cfg *RouterConfig) {
			cfg.ExternalModels[0].ModelEndpoint.Protocol = "ftp"
		}, want: "http or https"},
		{name: "missing external model name", mutate: func(cfg *RouterConfig) {
			cfg.ExternalModels[0].ModelName = ""
		}, want: "llm_model_name"},
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
	falseLegacyCfg := categoryBackendTestConfig(&RemoteClassifierBackend{
		Protocol: RemoteClassifierProtocolHTTPClassify,
		Model:    "named-category",
	})
	falseLegacyCfg.CategoryModel.UseModernBERT = false
	falseLegacyCfg.CategoryModel.UseMmBERT32K = false
	if err := ValidateCategoryModelBackend(falseLegacyCfg); err != nil {
		t.Fatalf("explicit false legacy selectors should remain readable, got %v", err)
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

func TestCanonicalCategoryVariantCompatibility(t *testing.T) {
	legacyOverride := []byte(`
version: v0.3
global:
  model_catalog:
    modules:
      classifier:
        domain:
          use_modernbert: true
`)
	cfg, err := ParseYAMLBytes(legacyOverride)
	if err != nil {
		t.Fatalf("legacy canonical override rejected: %v", err)
	}
	if got, want := cfg.CategoryModel.Variant, CategoryVariantModernBERT; got != want || cfg.CategoryModel.UseModernBERT {
		t.Fatalf("legacy canonical override = variant=%q modern=%v, want canonical modernbert without legacy spelling", got, cfg.CategoryModel.UseModernBERT)
	}

	conflictingOverride := []byte(`
version: v0.3
global:
  model_catalog:
    modules:
      classifier:
        domain:
          variant: modernbert
          use_mmbert_32k: true
`)
	if _, err := ParseYAMLBytes(conflictingOverride); err == nil || !strings.Contains(err.Error(), "conflicts") {
		t.Fatalf("expected explicit canonical/legacy conflict, got %v", err)
	}

	agreeingOverride := []byte(`
version: v0.3
global:
  model_catalog:
    modules:
      classifier:
        domain:
          variant: modernbert
          use_modernbert: true
`)
	agreeing, err := ParseYAMLBytes(agreeingOverride)
	if err != nil {
		t.Fatalf("agreeing canonical/legacy override rejected: %v", err)
	}
	if agreeing.CategoryModel.Variant != CategoryVariantModernBERT || agreeing.CategoryModel.UseModernBERT {
		t.Fatalf("agreeing override was not canonicalized: variant=%q legacy=%v", agreeing.CategoryModel.Variant, agreeing.CategoryModel.UseModernBERT)
	}

	clearingOverride := []byte(`
version: v0.3
global:
  model_catalog:
    modules:
      classifier:
        domain:
          use_mmbert_32k: false
`)
	cleared, err := ParseYAMLBytes(clearingOverride)
	if err != nil {
		t.Fatalf("legacy clear override rejected: %v", err)
	}
	if cleared.CategoryModel.Variant != "" || cleared.CategoryModel.UseMmBERT32K {
		t.Fatalf("legacy false override did not clear inherited variant: variant=%q legacy=%v", cleared.CategoryModel.Variant, cleared.CategoryModel.UseMmBERT32K)
	}
}

func TestCanonicalRemoteBackendReplacesInheritedLocalVariant(t *testing.T) {
	canonicalYAML := []byte(`
version: v0.3
global:
  model_catalog:
    external:
      - name: named-category
        model_role: classification
        llm_model_name: category-service
        llm_endpoint:
          address: 127.0.0.1
          port: 8080
    modules:
      classifier:
        domain:
          backend:
            protocol: http_classify
            contract: label_distribution.v1
            model: named-category
`)
	cfg, err := ParseYAMLBytes(canonicalYAML)
	if err != nil {
		t.Fatalf("remote canonical override rejected: %v", err)
	}
	if cfg.CategoryModel.Backend == nil || cfg.CategoryModel.Backend.Model != "named-category" {
		t.Fatalf("remote backend was not decoded: %#v", cfg.CategoryModel.Backend)
	}
	if cfg.CategoryModel.Backend.Contract != RemoteClassifierContractLabelDistribution {
		t.Fatalf("remote contract = %q, want %q", cfg.CategoryModel.Backend.Contract, RemoteClassifierContractLabelDistribution)
	}
	if cfg.CategoryModel.Variant != "" || cfg.CategoryModel.UseModernBERT || cfg.CategoryModel.UseMmBERT32K {
		t.Fatalf("inherited local selector was not cleared for remote backend: variant=%q modern=%v mmbert=%v",
			cfg.CategoryModel.Variant, cfg.CategoryModel.UseModernBERT, cfg.CategoryModel.UseMmBERT32K)
	}
}

func TestReferenceConfigCategoryBackendReplacesDefaultVariant(t *testing.T) {
	data := string(readReferenceConfigYAML(t))
	data = strings.Replace(data,
		"          category_mapping_path: models/mmbert32k-intent-classifier-merged/category_mapping.json\n",
		"          backend:\n"+
			"            protocol: http_classify\n"+
			"            contract: label_distribution.v1\n"+
			"            model: external-classifier\n"+
			"            deadline_ms: 5000\n"+
			"          category_mapping_path: models/mmbert32k-intent-classifier-merged/category_mapping.json\n", 1)
	if data == string(readReferenceConfigYAML(t)) {
		t.Fatal("reference config category block was not found")
	}
	if _, err := ParseYAMLBytes([]byte(data)); err != nil {
		t.Fatalf("reference config plus backend rejected: %v", err)
	}
}

func TestCanonicalRemoteBackendAllowsExplicitFalseLegacySelector(t *testing.T) {
	falseLegacyBackendYAML := []byte(`
version: v0.3
global:
  model_catalog:
    external:
      - name: named-category
        model_role: classification
        llm_model_name: category-service
        llm_endpoint:
          address: 127.0.0.1
          port: 8080
    modules:
      classifier:
        domain:
          use_mmbert_32k: false
          backend:
            protocol: http_classify
            model: named-category
`)
	falseLegacyCfg, err := ParseYAMLBytes(falseLegacyBackendYAML)
	if err != nil {
		t.Fatalf("remote backend with explicit false legacy selector rejected: %v", err)
	}
	if falseLegacyCfg.CategoryModel.Variant != "" || falseLegacyCfg.CategoryModel.UseMmBERT32K {
		t.Fatalf("explicit false legacy selector was not normalized for remote backend: variant=%q mmbert=%v",
			falseLegacyCfg.CategoryModel.Variant, falseLegacyCfg.CategoryModel.UseMmBERT32K)
	}
}

func TestCanonicalRemoteBackendRejectsActiveLegacySelector(t *testing.T) {
	trueLegacyBackendYAML := []byte(`
version: v0.3
global:
  model_catalog:
    external:
      - name: named-category
        model_role: classification
        llm_model_name: category-service
        llm_endpoint:
          address: 127.0.0.1
          port: 8080
    modules:
      classifier:
        domain:
          use_mmbert_32k: true
          backend:
            protocol: http_classify
            model: named-category
`)
	if _, err := ParseYAMLBytes(trueLegacyBackendYAML); err == nil || !strings.Contains(err.Error(), "backend is mutually exclusive") {
		t.Fatalf("expected backend with active legacy selector to be rejected, got %v", err)
	}
}

func TestCanonicalCategoryVariantExportUsesCanonicalSelector(t *testing.T) {
	cfg := &RouterConfig{
		InlineModels: InlineModels{Classifier: Classifier{CategoryModel: CategoryModel{
			Variant:       CategoryVariantModernBERT,
			UseModernBERT: true,
		}}},
	}
	exported := CanonicalGlobalFromRouterConfig(cfg)
	if exported == nil {
		t.Fatal("expected canonical global export")
	}
	domain := exported.ModelCatalog.Modules.Classifier.Domain.CategoryModel
	if domain.Variant != CategoryVariantModernBERT || domain.UseModernBERT || domain.UseMmBERT32K {
		t.Fatalf("exported category selector = variant=%q modern=%v mmbert=%v; want variant only", domain.Variant, domain.UseModernBERT, domain.UseMmBERT32K)
	}
	encoded, err := yaml.Marshal(exported.ModelCatalog.Modules.Classifier.Domain)
	if err != nil {
		t.Fatalf("marshal canonical export: %v", err)
	}
	if strings.Contains(string(encoded), "use_modernbert") || strings.Contains(string(encoded), "use_mmbert_32k") {
		t.Fatalf("canonical export retained deprecated selectors: %s", encoded)
	}
}
